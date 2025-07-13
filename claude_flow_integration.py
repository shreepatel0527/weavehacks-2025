"""
Claude-flow integration for advanced AI reasoning and planning
"""
import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import weave
from pathlib import Path
import numpy as np

class ReasoningType(Enum):
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"
    EXPLORATORY = "exploratory"
    SAFETY_CRITICAL = "safety_critical"

@dataclass
class ReasoningContext:
    experiment_state: Dict[str, Any]
    historical_data: List[Dict[str, Any]]
    sensor_readings: Dict[str, float]
    recent_events: List[Dict[str, Any]]
    safety_status: Dict[str, Any]
    protocol_step: Optional[Dict[str, Any]] = None
    user_query: Optional[str] = None

@dataclass
class ReasoningResult:
    reasoning_type: ReasoningType
    conclusion: str
    confidence: float
    recommendations: List[str]
    supporting_evidence: List[Dict[str, Any]]
    potential_risks: List[str]
    next_actions: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)

class ClaudeFlowReasoner:
    """Advanced reasoning system using Claude-flow capabilities"""
    
    def __init__(self):
        self.reasoning_history = []
        self.knowledge_base = self._load_knowledge_base()
        self.active_hypotheses = []
        self.learning_cache = {}
        
        # Initialize reasoning patterns
        self.reasoning_patterns = {
            ReasoningType.DIAGNOSTIC: self._diagnostic_reasoning,
            ReasoningType.PREDICTIVE: self._predictive_reasoning,
            ReasoningType.PRESCRIPTIVE: self._prescriptive_reasoning,
            ReasoningType.EXPLORATORY: self._exploratory_reasoning,
            ReasoningType.SAFETY_CRITICAL: self._safety_critical_reasoning
        }
        
        # Initialize W&B
        weave.init('claude-flow-reasoning')
    
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load domain knowledge for reasoning"""
        return {
            'chemical_properties': {
                'HAuCl4': {
                    'molecular_weight': 393.83,
                    'hazards': ['corrosive', 'oxidizing'],
                    'typical_concentration': 0.001  # M
                },
                'NaBH4': {
                    'molecular_weight': 37.83,
                    'hazards': ['flammable', 'water_reactive'],
                    'reduction_potential': -1.24  # V
                },
                'Au25': {
                    'typical_yield': 0.4,  # 40%
                    'size_nm': 1.2,
                    'stability_temp': 4  # °C
                }
            },
            'reaction_kinetics': {
                'color_transitions': {
                    'gold_to_yellow': {'duration': 300, 'temp_dependent': True},
                    'yellow_to_clear': {'duration': 3600, 'catalyst': 'PhCH2CH2SH'}
                },
                'critical_parameters': {
                    'temperature': {'min': 0, 'max': 25, 'optimal': 4},
                    'ph': {'min': 6, 'max': 8, 'optimal': 7},
                    'stirring_rpm': {'min': 500, 'max': 1500, 'optimal': 1100}
                }
            },
            'safety_thresholds': {
                'temperature_spike': 5,  # °C/min
                'pressure_increase': 10,  # kPa/min
                'gas_evolution_rate': 100  # mL/min
            }
        }
    
    @weave.op()
    async def reason(self, context: ReasoningContext, 
                    reasoning_type: Optional[ReasoningType] = None) -> ReasoningResult:
        """Perform advanced reasoning based on context"""
        
        # Determine reasoning type if not specified
        if reasoning_type is None:
            reasoning_type = self._determine_reasoning_type(context)
        
        # Log reasoning request
        weave.log({
            'reasoning_request': {
                'type': reasoning_type.value,
                'has_user_query': context.user_query is not None,
                'sensor_count': len(context.sensor_readings),
                'event_count': len(context.recent_events),
                'timestamp': datetime.now().isoformat()
            }
        })
        
        # Execute appropriate reasoning pattern
        reasoning_func = self.reasoning_patterns.get(reasoning_type)
        if reasoning_func:
            result = await reasoning_func(context)
        else:
            result = await self._default_reasoning(context)
        
        # Store in history
        self.reasoning_history.append({
            'context': context,
            'result': result,
            'timestamp': datetime.now()
        })
        
        # Update learning cache
        self._update_learning(context, result)
        
        return result
    
    def _determine_reasoning_type(self, context: ReasoningContext) -> ReasoningType:
        """Determine appropriate reasoning type from context"""
        
        # Check for safety issues first
        if context.safety_status and not context.safety_status.get('is_safe', True):
            return ReasoningType.SAFETY_CRITICAL
        
        # Check for diagnostic needs
        if any(event.get('type') == 'anomaly' for event in context.recent_events):
            return ReasoningType.DIAGNOSTIC
        
        # Check for user query
        if context.user_query:
            if 'what will' in context.user_query.lower() or 'predict' in context.user_query.lower():
                return ReasoningType.PREDICTIVE
            elif 'should i' in context.user_query.lower() or 'recommend' in context.user_query.lower():
                return ReasoningType.PRESCRIPTIVE
            else:
                return ReasoningType.EXPLORATORY
        
        # Default to predictive
        return ReasoningType.PREDICTIVE
    
    async def _diagnostic_reasoning(self, context: ReasoningContext) -> ReasoningResult:
        """Diagnose issues and anomalies"""
        
        issues_found = []
        recommendations = []
        evidence = []
        
        # Analyze recent events for patterns
        event_types = [e.get('type') for e in context.recent_events]
        
        # Check for unexpected color changes
        color_events = [e for e in context.recent_events if e.get('type') == 'color_change']
        if color_events:
            last_color = color_events[-1].get('color')
            expected_color = self._get_expected_color(context.protocol_step)
            
            if last_color != expected_color:
                issues_found.append(f"Unexpected color: {last_color} (expected {expected_color})")
                evidence.append({
                    'type': 'color_mismatch',
                    'observed': last_color,
                    'expected': expected_color
                })
                recommendations.append("Check reagent quality and concentrations")
        
        # Check reaction kinetics
        if context.protocol_step:
            step_duration = self._calculate_step_duration(context)
            expected_duration = self.knowledge_base['reaction_kinetics'].get(
                'color_transitions', {}
            ).get(context.protocol_step.get('name', ''), {}).get('duration', 0)
            
            if expected_duration and abs(step_duration - expected_duration) > 300:  # 5 min tolerance
                issues_found.append(f"Reaction time deviation: {step_duration}s (expected {expected_duration}s)")
                evidence.append({
                    'type': 'kinetics_deviation',
                    'observed_duration': step_duration,
                    'expected_duration': expected_duration
                })
                recommendations.append("Verify temperature and stirring speed")
        
        # Analyze sensor readings
        sensor_issues = self._analyze_sensor_readings(context.sensor_readings)
        issues_found.extend(sensor_issues['issues'])
        evidence.extend(sensor_issues['evidence'])
        recommendations.extend(sensor_issues['recommendations'])
        
        # Generate conclusion
        if issues_found:
            conclusion = f"Diagnostic analysis found {len(issues_found)} issues: " + "; ".join(issues_found[:2])
            confidence = 0.8
        else:
            conclusion = "No significant issues detected. Experiment proceeding normally."
            confidence = 0.9
        
        return ReasoningResult(
            reasoning_type=ReasoningType.DIAGNOSTIC,
            conclusion=conclusion,
            confidence=confidence,
            recommendations=recommendations,
            supporting_evidence=evidence,
            potential_risks=self._assess_risks(context, issues_found),
            next_actions=self._suggest_diagnostic_actions(issues_found)
        )
    
    async def _predictive_reasoning(self, context: ReasoningContext) -> ReasoningResult:
        """Predict future experiment outcomes"""
        
        predictions = []
        evidence = []
        
        # Predict yield based on current parameters
        predicted_yield = self._predict_yield(context)
        predictions.append(f"Expected yield: {predicted_yield['yield']:.1f}% (±{predicted_yield['uncertainty']:.1f}%)")
        evidence.append({
            'type': 'yield_prediction',
            'model': 'kinetic_regression',
            'inputs': predicted_yield['factors']
        })
        
        # Predict completion time
        time_prediction = self._predict_completion_time(context)
        predictions.append(f"Estimated completion: {time_prediction['time']} minutes from now")
        evidence.append({
            'type': 'time_prediction',
            'based_on': time_prediction['factors']
        })
        
        # Predict potential issues
        risk_predictions = self._predict_risks(context)
        if risk_predictions:
            predictions.append(f"Potential risks: {', '.join(r['risk'] for r in risk_predictions[:2])}")
            evidence.extend(risk_predictions)
        
        # Generate recommendations
        recommendations = self._generate_predictive_recommendations(
            predicted_yield, time_prediction, risk_predictions
        )
        
        conclusion = f"Prediction: {predictions[0]}. {predictions[1] if len(predictions) > 1 else ''}"
        
        return ReasoningResult(
            reasoning_type=ReasoningType.PREDICTIVE,
            conclusion=conclusion,
            confidence=predicted_yield['confidence'],
            recommendations=recommendations,
            supporting_evidence=evidence,
            potential_risks=[r['risk'] for r in risk_predictions],
            next_actions=self._suggest_predictive_actions(predictions)
        )
    
    async def _prescriptive_reasoning(self, context: ReasoningContext) -> ReasoningResult:
        """Prescribe optimal actions"""
        
        prescriptions = []
        evidence = []
        
        # Analyze current state
        state_analysis = self._analyze_experiment_state(context)
        
        # Generate prescriptions based on state
        if state_analysis['phase'] == 'synthesis':
            prescriptions.extend(self._prescribe_synthesis_actions(context, state_analysis))
        elif state_analysis['phase'] == 'purification':
            prescriptions.extend(self._prescribe_purification_actions(context, state_analysis))
        else:
            prescriptions.extend(self._prescribe_general_actions(context, state_analysis))
        
        # Prioritize prescriptions
        prioritized = self._prioritize_prescriptions(prescriptions, context)
        
        # Generate evidence
        for prescription in prioritized[:3]:
            evidence.append({
                'type': 'prescription_rationale',
                'action': prescription['action'],
                'expected_outcome': prescription['expected_outcome'],
                'confidence': prescription['confidence']
            })
        
        conclusion = f"Recommended action: {prioritized[0]['action']}" if prioritized else "Continue with current protocol"
        
        return ReasoningResult(
            reasoning_type=ReasoningType.PRESCRIPTIVE,
            conclusion=conclusion,
            confidence=prioritized[0]['confidence'] if prioritized else 0.7,
            recommendations=[p['action'] for p in prioritized[:5]],
            supporting_evidence=evidence,
            potential_risks=self._assess_prescription_risks(prioritized),
            next_actions=[self._convert_to_action(p) for p in prioritized[:3]]
        )
    
    async def _exploratory_reasoning(self, context: ReasoningContext) -> ReasoningResult:
        """Explore possibilities and answer queries"""
        
        query_response = ""
        evidence = []
        recommendations = []
        
        if context.user_query:
            # Parse query intent
            query_intent = self._parse_query_intent(context.user_query)
            
            # Generate response based on intent
            if query_intent['type'] == 'explanation':
                query_response = self._generate_explanation(query_intent['topic'], context)
            elif query_intent['type'] == 'comparison':
                query_response = self._generate_comparison(query_intent['items'], context)
            elif query_intent['type'] == 'hypothesis':
                query_response = self._evaluate_hypothesis(query_intent['hypothesis'], context)
            else:
                query_response = self._general_exploration(context.user_query, context)
            
            # Add supporting evidence
            evidence = self._gather_exploration_evidence(query_intent, context)
            
            # Generate follow-up recommendations
            recommendations = self._suggest_explorations(query_intent, context)
        
        conclusion = query_response or "Exploration complete. See recommendations for insights."
        
        return ReasoningResult(
            reasoning_type=ReasoningType.EXPLORATORY,
            conclusion=conclusion,
            confidence=0.75,
            recommendations=recommendations,
            supporting_evidence=evidence,
            potential_risks=[],
            next_actions=self._suggest_exploration_actions(recommendations)
        )
    
    async def _safety_critical_reasoning(self, context: ReasoningContext) -> ReasoningResult:
        """Handle safety-critical situations"""
        
        # Assess immediate dangers
        dangers = self._assess_immediate_dangers(context)
        
        # Generate emergency actions
        emergency_actions = []
        for danger in dangers:
            actions = self._generate_emergency_actions(danger)
            emergency_actions.extend(actions)
        
        # Prioritize by severity
        prioritized_actions = sorted(
            emergency_actions, 
            key=lambda x: x['priority'], 
            reverse=True
        )
        
        # Generate evidence
        evidence = []
        for danger in dangers:
            evidence.append({
                'type': 'safety_violation',
                'parameter': danger['parameter'],
                'value': danger['value'],
                'threshold': danger['threshold'],
                'severity': danger['severity']
            })
        
        # Assess cascade risks
        cascade_risks = self._assess_cascade_risks(dangers, context)
        
        conclusion = f"SAFETY CRITICAL: {len(dangers)} immediate dangers detected. " + \
                    f"Execute emergency protocol: {prioritized_actions[0]['action']}" if prioritized_actions else ""
        
        return ReasoningResult(
            reasoning_type=ReasoningType.SAFETY_CRITICAL,
            conclusion=conclusion,
            confidence=0.95,
            recommendations=[a['action'] for a in prioritized_actions],
            supporting_evidence=evidence,
            potential_risks=cascade_risks,
            next_actions=prioritized_actions[:5]
        )
    
    # Helper methods
    
    def _analyze_sensor_readings(self, readings: Dict[str, float]) -> Dict[str, Any]:
        """Analyze sensor readings for issues"""
        issues = []
        evidence = []
        recommendations = []
        
        critical_params = self.knowledge_base['reaction_kinetics']['critical_parameters']
        
        for param, value in readings.items():
            if param in critical_params:
                limits = critical_params[param]
                
                if value < limits['min'] or value > limits['max']:
                    issues.append(f"{param} out of range: {value}")
                    evidence.append({
                        'type': 'parameter_violation',
                        'parameter': param,
                        'value': value,
                        'limits': limits
                    })
                    
                    if value < limits['min']:
                        recommendations.append(f"Increase {param} to at least {limits['min']}")
                    else:
                        recommendations.append(f"Reduce {param} to below {limits['max']}")
        
        return {
            'issues': issues,
            'evidence': evidence,
            'recommendations': recommendations
        }
    
    def _predict_yield(self, context: ReasoningContext) -> Dict[str, Any]:
        """Predict reaction yield based on current conditions"""
        
        # Simplified yield prediction model
        base_yield = self.knowledge_base['chemical_properties']['Au25']['typical_yield']
        
        # Factors affecting yield
        factors = {}
        yield_modifier = 1.0
        
        # Temperature effect
        if 'temperature' in context.sensor_readings:
            temp = context.sensor_readings['temperature']
            optimal_temp = self.knowledge_base['reaction_kinetics']['critical_parameters']['temperature']['optimal']
            temp_deviation = abs(temp - optimal_temp)
            temp_factor = 1.0 - (temp_deviation * 0.02)  # 2% reduction per degree
            yield_modifier *= temp_factor
            factors['temperature'] = {'value': temp, 'factor': temp_factor}
        
        # Stirring effect
        if 'stirring_rpm' in context.sensor_readings:
            rpm = context.sensor_readings['stirring_rpm']
            optimal_rpm = self.knowledge_base['reaction_kinetics']['critical_parameters']['stirring_rpm']['optimal']
            rpm_factor = 1.0 - (abs(rpm - optimal_rpm) / optimal_rpm * 0.1)
            yield_modifier *= rpm_factor
            factors['stirring'] = {'value': rpm, 'factor': rpm_factor}
        
        # Calculate final yield
        predicted_yield = base_yield * yield_modifier * 100
        uncertainty = 5.0 * (2.0 - yield_modifier)  # Higher uncertainty with worse conditions
        
        return {
            'yield': predicted_yield,
            'uncertainty': uncertainty,
            'confidence': 0.7 + (0.2 * yield_modifier),
            'factors': factors
        }
    
    def _predict_completion_time(self, context: ReasoningContext) -> Dict[str, Any]:
        """Predict time to completion"""
        
        # Get current step
        current_step = context.protocol_step
        if not current_step:
            return {'time': 'unknown', 'factors': ['no_protocol_info']}
        
        # Estimate remaining time
        total_remaining = 0
        factors = []
        
        # Time for current step
        if current_step.get('duration'):
            elapsed = (datetime.now() - current_step.get('start_time', datetime.now())).total_seconds()
            remaining_current = max(0, current_step['duration'] - elapsed)
            total_remaining += remaining_current
            factors.append(f"current_step: {remaining_current/60:.1f}min")
        
        # Time for remaining steps (simplified)
        remaining_steps = current_step.get('remaining_steps', 5)
        avg_step_time = 600  # 10 minutes average
        total_remaining += remaining_steps * avg_step_time
        factors.append(f"remaining_steps: {remaining_steps}")
        
        return {
            'time': int(total_remaining / 60),
            'factors': factors
        }
    
    def _update_learning(self, context: ReasoningContext, result: ReasoningResult):
        """Update learning cache with new insights"""
        
        # Store successful predictions
        if result.confidence > 0.8:
            cache_key = f"{result.reasoning_type.value}_{context.protocol_step.get('name', 'unknown')}"
            
            self.learning_cache[cache_key] = {
                'context_summary': {
                    'sensors': dict(context.sensor_readings),
                    'events': len(context.recent_events)
                },
                'result_summary': {
                    'conclusion': result.conclusion,
                    'confidence': result.confidence
                },
                'timestamp': datetime.now()
            }
            
            # Limit cache size
            if len(self.learning_cache) > 100:
                oldest_key = min(self.learning_cache.keys(), 
                               key=lambda k: self.learning_cache[k]['timestamp'])
                del self.learning_cache[oldest_key]

class ClaudeFlowOrchestrator:
    """Orchestrate complex reasoning workflows"""
    
    def __init__(self, reasoner: ClaudeFlowReasoner):
        self.reasoner = reasoner
        self.workflow_history = []
        
    async def orchestrate_experiment_analysis(self, 
                                            full_context: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrate complete experiment analysis"""
        
        results = {}
        
        # Create reasoning context
        context = ReasoningContext(
            experiment_state=full_context.get('experiment_state', {}),
            historical_data=full_context.get('historical_data', []),
            sensor_readings=full_context.get('sensor_readings', {}),
            recent_events=full_context.get('recent_events', []),
            safety_status=full_context.get('safety_status', {}),
            protocol_step=full_context.get('current_step')
        )
        
        # Check safety first
        safety_result = await self.reasoner.reason(context, ReasoningType.SAFETY_CRITICAL)
        results['safety'] = safety_result
        
        # If safe, continue with other analyses
        if safety_result.confidence < 0.9 or not safety_result.potential_risks:
            # Diagnostic reasoning
            diagnostic_result = await self.reasoner.reason(context, ReasoningType.DIAGNOSTIC)
            results['diagnostic'] = diagnostic_result
            
            # Predictive reasoning
            predictive_result = await self.reasoner.reason(context, ReasoningType.PREDICTIVE)
            results['predictive'] = predictive_result
            
            # Prescriptive reasoning
            prescriptive_result = await self.reasoner.reason(context, ReasoningType.PRESCRIPTIVE)
            results['prescriptive'] = prescriptive_result
        
        # Synthesize results
        synthesis = self._synthesize_results(results)
        
        # Store in history
        self.workflow_history.append({
            'timestamp': datetime.now(),
            'context': full_context,
            'results': results,
            'synthesis': synthesis
        })
        
        return synthesis
    
    def _synthesize_results(self, results: Dict[str, ReasoningResult]) -> Dict[str, Any]:
        """Synthesize multiple reasoning results"""
        
        # Prioritize actions
        all_actions = []
        for result in results.values():
            all_actions.extend(result.next_actions)
        
        # Remove duplicates and prioritize
        unique_actions = []
        seen = set()
        for action in sorted(all_actions, key=lambda x: x.get('priority', 0), reverse=True):
            action_key = action.get('action', '')
            if action_key not in seen:
                seen.add(action_key)
                unique_actions.append(action)
        
        # Generate overall assessment
        overall_confidence = np.mean([r.confidence for r in results.values()])
        
        return {
            'overall_status': self._determine_overall_status(results),
            'confidence': overall_confidence,
            'priority_actions': unique_actions[:5],
            'key_insights': self._extract_key_insights(results),
            'recommendations': self._consolidate_recommendations(results)
        }
    
    def _determine_overall_status(self, results: Dict[str, ReasoningResult]) -> str:
        """Determine overall experiment status"""
        
        if 'safety' in results and results['safety'].potential_risks:
            return 'critical'
        elif any(r.confidence < 0.5 for r in results.values()):
            return 'concerning'
        elif all(r.confidence > 0.8 for r in results.values()):
            return 'excellent'
        else:
            return 'normal'
    
    def _extract_key_insights(self, results: Dict[str, ReasoningResult]) -> List[str]:
        """Extract key insights from all reasoning results"""
        
        insights = []
        
        for reasoning_type, result in results.items():
            if result.confidence > 0.7:
                insights.append(f"{reasoning_type}: {result.conclusion}")
        
        return insights[:5]
    
    def _consolidate_recommendations(self, results: Dict[str, ReasoningResult]) -> List[str]:
        """Consolidate recommendations from all reasoning"""
        
        all_recommendations = []
        
        for result in results.values():
            all_recommendations.extend(result.recommendations)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        
        for rec in all_recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:10]

# Example usage
async def demo_claude_flow_reasoning():
    """Demonstrate Claude-flow reasoning capabilities"""
    
    # Create reasoner
    reasoner = ClaudeFlowReasoner()
    orchestrator = ClaudeFlowOrchestrator(reasoner)
    
    # Sample context
    sample_context = {
        'experiment_state': {
            'phase': 'synthesis',
            'step_number': 5,
            'elapsed_time': 1800
        },
        'sensor_readings': {
            'temperature': 23.5,
            'pressure': 101.3,
            'stirring_rpm': 1050,
            'ph': 7.2
        },
        'recent_events': [
            {'type': 'color_change', 'color': 'yellow', 'timestamp': datetime.now()},
            {'type': 'temperature_spike', 'value': 28, 'timestamp': datetime.now()}
        ],
        'safety_status': {
            'is_safe': True,
            'warnings': []
        },
        'current_step': {
            'name': 'gold_reduction',
            'duration': 3600,
            'start_time': datetime.now()
        }
    }
    
    # Run orchestrated analysis
    print("Running Claude-flow experiment analysis...")
    analysis_result = await orchestrator.orchestrate_experiment_analysis(sample_context)
    
    print(f"\nOverall Status: {analysis_result['overall_status']}")
    print(f"Confidence: {analysis_result['confidence']:.2f}")
    
    print("\nKey Insights:")
    for insight in analysis_result['key_insights']:
        print(f"  • {insight}")
    
    print("\nPriority Actions:")
    for action in analysis_result['priority_actions']:
        print(f"  • {action.get('action', 'Unknown action')}")
    
    print("\nRecommendations:")
    for rec in analysis_result['recommendations'][:5]:
        print(f"  • {rec}")

if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_claude_flow_reasoning())