"""
Predictive ML models for experiment optimization and outcome prediction
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import weave
from pathlib import Path

@dataclass
class ExperimentFeatures:
    """Features for ML prediction"""
    # Reagent quantities
    gold_mass: float
    toab_mass: float
    sulfur_mass: float
    nabh4_mass: float
    
    # Process parameters
    temperature: float
    stirring_rpm: int
    reaction_time: float
    ph: float
    
    # Environmental conditions
    ambient_temp: float
    humidity: float
    pressure: float
    
    # Derived features
    gold_to_sulfur_ratio: float = 0.0
    gold_to_nabh4_ratio: float = 0.0
    temperature_stability: float = 0.0
    
    def __post_init__(self):
        """Calculate derived features"""
        if self.sulfur_mass > 0:
            self.gold_to_sulfur_ratio = self.gold_mass / self.sulfur_mass
        if self.nabh4_mass > 0:
            self.gold_to_nabh4_ratio = self.gold_mass / self.nabh4_mass

@dataclass
class PredictionResult:
    """Result from ML prediction"""
    prediction: float
    confidence_interval: Tuple[float, float]
    feature_importance: Dict[str, float]
    model_type: str
    timestamp: datetime = field(default_factory=datetime.now)

class YieldPredictor:
    """Predict nanoparticle synthesis yield"""
    
    def __init__(self, model_path: Optional[Path] = None):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.training_history = []
        
        if model_path and model_path.exists():
            self.load_models(model_path)
        else:
            self._initialize_models()
        
        weave.init('yield-predictor')
    
    def _initialize_models(self):
        """Initialize ML models"""
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ),
            'nn': MLPRegressor(
                hidden_layer_sizes=(50, 30),
                activation='relu',
                solver='adam',
                random_state=42
            )
        }
        
        self.scalers = {
            'rf': StandardScaler(),
            'gb': StandardScaler(),
            'nn': StandardScaler()
        }
    
    @weave.op()
    def train(self, training_data: pd.DataFrame, target_column: str = 'yield'):
        """Train models on historical data"""
        
        # Prepare features and target
        feature_columns = [col for col in training_data.columns if col != target_column]
        self.feature_names = feature_columns
        
        X = training_data[feature_columns].values
        y = training_data[target_column].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train each model
        results = {}
        
        for model_name, model in self.models.items():
            # Scale features
            scaler = self.scalers[model_name]
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=5, scoring='r2'
            )
            
            results[model_name] = {
                'train_r2': train_score,
                'test_r2': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Log training
            weave.log({
                'model_training': {
                    'model': model_name,
                    'train_r2': train_score,
                    'test_r2': test_score,
                    'cv_mean': cv_scores.mean(),
                    'n_samples': len(X_train)
                }
            })
        
        # Store training history
        self.training_history.append({
            'timestamp': datetime.now(),
            'n_samples': len(X),
            'results': results
        })
        
        return results
    
    @weave.op()
    def predict(self, features: ExperimentFeatures, 
                model_type: str = 'ensemble') -> PredictionResult:
        """Predict yield for given features"""
        
        # Convert features to array
        feature_array = self._features_to_array(features)
        
        if model_type == 'ensemble':
            # Ensemble prediction
            predictions = []
            importances = []
            
            for model_name, model in self.models.items():
                # Scale features
                scaled_features = self.scalers[model_name].transform([feature_array])
                
                # Predict
                pred = model.predict(scaled_features)[0]
                predictions.append(pred)
                
                # Get feature importance if available
                if hasattr(model, 'feature_importances_'):
                    importances.append(model.feature_importances_)
            
            # Ensemble prediction (weighted average)
            weights = [0.4, 0.4, 0.2]  # RF, GB, NN weights
            ensemble_prediction = np.average(predictions, weights=weights)
            
            # Confidence interval (simplified)
            std_dev = np.std(predictions)
            confidence_interval = (
                ensemble_prediction - 2 * std_dev,
                ensemble_prediction + 2 * std_dev
            )
            
            # Average feature importance
            if importances:
                avg_importance = np.mean(importances[:2], axis=0)  # RF and GB only
                feature_importance = dict(zip(self.feature_names, avg_importance))
            else:
                feature_importance = {}
            
        else:
            # Single model prediction
            model = self.models.get(model_type)
            scaler = self.scalers.get(model_type)
            
            if not model or not scaler:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Scale and predict
            scaled_features = scaler.transform([feature_array])
            prediction = model.predict(scaled_features)[0]
            
            # Simple confidence interval
            confidence_interval = (prediction * 0.9, prediction * 1.1)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(self.feature_names, model.feature_importances_))
            else:
                feature_importance = {}
        
        # Log prediction
        weave.log({
            'yield_prediction': {
                'prediction': ensemble_prediction if model_type == 'ensemble' else prediction,
                'model_type': model_type,
                'confidence_interval': confidence_interval,
                'features': features.__dict__
            }
        })
        
        return PredictionResult(
            prediction=ensemble_prediction if model_type == 'ensemble' else prediction,
            confidence_interval=confidence_interval,
            feature_importance=feature_importance,
            model_type=model_type
        )
    
    def _features_to_array(self, features: ExperimentFeatures) -> np.ndarray:
        """Convert features to array for model input"""
        return np.array([
            features.gold_mass,
            features.toab_mass,
            features.sulfur_mass,
            features.nabh4_mass,
            features.temperature,
            features.stirring_rpm,
            features.reaction_time,
            features.ph,
            features.ambient_temp,
            features.humidity,
            features.pressure,
            features.gold_to_sulfur_ratio,
            features.gold_to_nabh4_ratio,
            features.temperature_stability
        ])
    
    def save_models(self, path: Path):
        """Save trained models"""
        path.mkdir(exist_ok=True)
        
        for model_name, model in self.models.items():
            joblib.dump(model, path / f"{model_name}_model.pkl")
            joblib.dump(self.scalers[model_name], path / f"{model_name}_scaler.pkl")
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }
        
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, default=str)
    
    def load_models(self, path: Path):
        """Load saved models"""
        for model_name in ['rf', 'gb', 'nn']:
            model_path = path / f"{model_name}_model.pkl"
            scaler_path = path / f"{model_name}_scaler.pkl"
            
            if model_path.exists() and scaler_path.exists():
                self.models[model_name] = joblib.load(model_path)
                self.scalers[model_name] = joblib.load(scaler_path)
        
        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.feature_names = metadata.get('feature_names', [])
                self.training_history = metadata.get('training_history', [])

class ReactionOptimizer:
    """Optimize reaction conditions for maximum yield"""
    
    def __init__(self, yield_predictor: YieldPredictor):
        self.predictor = yield_predictor
        self.optimization_history = []
        weave.init('reaction-optimizer')
    
    @weave.op()
    def optimize(self, 
                 base_features: ExperimentFeatures,
                 optimization_targets: Dict[str, Tuple[float, float]],
                 n_iterations: int = 50) -> Dict[str, Any]:
        """Optimize reaction conditions"""
        
        best_features = base_features
        best_yield = 0.0
        
        # Parameter ranges
        param_ranges = {
            'temperature': (15, 30),
            'stirring_rpm': (800, 1200),
            'ph': (6.5, 7.5),
            'gold_to_sulfur_ratio': (0.8, 1.2),
            'gold_to_nabh4_ratio': (0.08, 0.12)
        }
        
        # Update with custom targets
        param_ranges.update(optimization_targets)
        
        # Optimization loop
        for i in range(n_iterations):
            # Generate candidate
            candidate = self._generate_candidate(best_features, param_ranges)
            
            # Predict yield
            result = self.predictor.predict(candidate)
            
            # Update best if improved
            if result.prediction > best_yield:
                best_yield = result.prediction
                best_features = candidate
                
                # Log improvement
                weave.log({
                    'optimization_improvement': {
                        'iteration': i,
                        'yield': best_yield,
                        'features': candidate.__dict__
                    }
                })
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            base_features, best_features, best_yield
        )
        
        # Store in history
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'base_yield': self.predictor.predict(base_features).prediction,
            'optimized_yield': best_yield,
            'improvement': best_yield - self.predictor.predict(base_features).prediction,
            'recommendations': recommendations
        })
        
        return {
            'optimized_features': best_features,
            'predicted_yield': best_yield,
            'improvement': best_yield - self.predictor.predict(base_features).prediction,
            'recommendations': recommendations
        }
    
    def _generate_candidate(self, 
                           base: ExperimentFeatures,
                           ranges: Dict[str, Tuple[float, float]]) -> ExperimentFeatures:
        """Generate candidate features for optimization"""
        
        # Copy base features
        candidate_dict = base.__dict__.copy()
        
        # Randomly modify some parameters
        n_params_to_modify = np.random.randint(1, 4)
        params_to_modify = np.random.choice(
            list(ranges.keys()), 
            size=n_params_to_modify, 
            replace=False
        )
        
        for param in params_to_modify:
            if param in candidate_dict:
                min_val, max_val = ranges[param]
                # Small perturbation
                current_val = candidate_dict[param]
                perturbation = np.random.normal(0, (max_val - min_val) * 0.1)
                new_val = np.clip(current_val + perturbation, min_val, max_val)
                candidate_dict[param] = new_val
        
        return ExperimentFeatures(**candidate_dict)
    
    def _generate_recommendations(self,
                                 base: ExperimentFeatures,
                                 optimized: ExperimentFeatures,
                                 predicted_yield: float) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # Compare parameters
        for param in ['temperature', 'stirring_rpm', 'ph']:
            base_val = getattr(base, param)
            opt_val = getattr(optimized, param)
            
            if abs(opt_val - base_val) > 0.01:
                if opt_val > base_val:
                    recommendations.append(
                        f"Increase {param} from {base_val:.1f} to {opt_val:.1f}"
                    )
                else:
                    recommendations.append(
                        f"Decrease {param} from {base_val:.1f} to {opt_val:.1f}"
                    )
        
        # Add yield improvement
        recommendations.append(
            f"Expected yield improvement: {predicted_yield:.1f}%"
        )
        
        return recommendations

class AnomalyDetector:
    """Detect anomalies in experiment data"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.baseline_stats = {}
        self.anomaly_history = []
        weave.init('anomaly-detector')
    
    def establish_baseline(self, historical_data: pd.DataFrame):
        """Establish baseline statistics"""
        
        for column in historical_data.columns:
            self.baseline_stats[column] = {
                'mean': historical_data[column].mean(),
                'std': historical_data[column].std(),
                'min': historical_data[column].min(),
                'max': historical_data[column].max()
            }
    
    @weave.op()
    def detect_anomalies(self, 
                        current_data: Dict[str, float],
                        sensitivity: float = 2.0) -> List[Dict[str, Any]]:
        """Detect anomalies in current data"""
        
        anomalies = []
        
        for param, value in current_data.items():
            if param in self.baseline_stats:
                stats = self.baseline_stats[param]
                
                # Z-score based detection
                z_score = abs((value - stats['mean']) / stats['std']) if stats['std'] > 0 else 0
                
                if z_score > sensitivity:
                    anomaly = {
                        'parameter': param,
                        'value': value,
                        'z_score': z_score,
                        'expected_range': (
                            stats['mean'] - sensitivity * stats['std'],
                            stats['mean'] + sensitivity * stats['std']
                        ),
                        'severity': 'high' if z_score > 3 else 'medium',
                        'timestamp': datetime.now()
                    }
                    anomalies.append(anomaly)
                    
                    # Log anomaly
                    weave.log({
                        'anomaly_detected': {
                            'parameter': param,
                            'value': value,
                            'z_score': z_score,
                            'severity': anomaly['severity']
                        }
                    })
        
        # Store in history
        if anomalies:
            self.anomaly_history.extend(anomalies)
        
        return anomalies

class ExperimentForecaster:
    """Forecast experiment outcomes and timeline"""
    
    def __init__(self):
        self.forecast_models = {}
        self.forecast_history = []
        weave.init('experiment-forecaster')
    
    @weave.op()
    def forecast_completion(self, 
                           current_step: int,
                           total_steps: int,
                           step_durations: List[float]) -> Dict[str, Any]:
        """Forecast experiment completion time"""
        
        # Calculate average step duration
        avg_duration = np.mean(step_durations) if step_durations else 600  # 10 min default
        
        # Estimate remaining time
        remaining_steps = total_steps - current_step
        estimated_time = remaining_steps * avg_duration
        
        # Add buffer for variability
        time_buffer = estimated_time * 0.2
        
        # Generate forecast
        forecast = {
            'estimated_completion': datetime.now() + timedelta(seconds=estimated_time),
            'confidence_interval': (
                datetime.now() + timedelta(seconds=estimated_time - time_buffer),
                datetime.now() + timedelta(seconds=estimated_time + time_buffer)
            ),
            'remaining_steps': remaining_steps,
            'average_step_duration': avg_duration,
            'progress_percentage': (current_step / total_steps) * 100
        }
        
        # Log forecast
        weave.log({
            'completion_forecast': {
                'current_step': current_step,
                'remaining_time_minutes': estimated_time / 60,
                'confidence': 0.8 if len(step_durations) > 5 else 0.6
            }
        })
        
        return forecast
    
    def forecast_parameter_trends(self,
                                 parameter_history: pd.DataFrame,
                                 forecast_horizon: int = 10) -> Dict[str, Any]:
        """Forecast parameter trends"""
        
        forecasts = {}
        
        for column in parameter_history.columns:
            if column == 'timestamp':
                continue
            
            # Simple linear extrapolation
            values = parameter_history[column].values[-10:]  # Last 10 points
            
            if len(values) > 2:
                # Fit linear trend
                x = np.arange(len(values))
                coeffs = np.polyfit(x, values, 1)
                
                # Forecast
                future_x = np.arange(len(values), len(values) + forecast_horizon)
                forecast_values = np.polyval(coeffs, future_x)
                
                forecasts[column] = {
                    'trend': 'increasing' if coeffs[0] > 0 else 'decreasing',
                    'rate': coeffs[0],
                    'forecast': forecast_values.tolist(),
                    'current': values[-1]
                }
        
        return forecasts

# Example usage
def demo_predictive_models():
    """Demonstrate predictive modeling capabilities"""
    
    # Generate sample training data
    np.random.seed(42)
    n_samples = 100
    
    training_data = pd.DataFrame({
        'gold_mass': np.random.uniform(0.15, 0.17, n_samples),
        'toab_mass': np.random.uniform(0.24, 0.26, n_samples),
        'sulfur_mass': np.random.uniform(0.05, 0.055, n_samples),
        'nabh4_mass': np.random.uniform(0.014, 0.016, n_samples),
        'temperature': np.random.uniform(20, 25, n_samples),
        'stirring_rpm': np.random.uniform(1000, 1200, n_samples),
        'reaction_time': np.random.uniform(3000, 4000, n_samples),
        'ph': np.random.uniform(6.8, 7.2, n_samples),
        'ambient_temp': np.random.uniform(20, 23, n_samples),
        'humidity': np.random.uniform(40, 60, n_samples),
        'pressure': np.random.uniform(100, 102, n_samples),
        'yield': np.random.uniform(35, 45, n_samples)  # Target variable
    })
    
    # Create and train yield predictor
    print("Training yield prediction models...")
    predictor = YieldPredictor()
    training_results = predictor.train(training_data)
    
    print("\nTraining Results:")
    for model, results in training_results.items():
        print(f"{model}: R² = {results['test_r2']:.3f} (CV: {results['cv_mean']:.3f} ± {results['cv_std']:.3f})")
    
    # Make a prediction
    test_features = ExperimentFeatures(
        gold_mass=0.1576,
        toab_mass=0.25,
        sulfur_mass=0.052,
        nabh4_mass=0.015,
        temperature=23.0,
        stirring_rpm=1100,
        reaction_time=3600,
        ph=7.0,
        ambient_temp=22.0,
        humidity=50.0,
        pressure=101.3
    )
    
    prediction = predictor.predict(test_features)
    print(f"\nPredicted Yield: {prediction.prediction:.1f}% "
          f"(CI: {prediction.confidence_interval[0]:.1f}-{prediction.confidence_interval[1]:.1f}%)")
    
    # Optimize conditions
    print("\nOptimizing reaction conditions...")
    optimizer = ReactionOptimizer(predictor)
    optimization_result = optimizer.optimize(test_features, {}, n_iterations=20)
    
    print(f"Optimization Result: {optimization_result['predicted_yield']:.1f}% yield")
    print("Recommendations:")
    for rec in optimization_result['recommendations']:
        print(f"  • {rec}")
    
    # Detect anomalies
    print("\nTesting anomaly detection...")
    detector = AnomalyDetector()
    detector.establish_baseline(training_data)
    
    anomalous_data = {
        'temperature': 35.0,  # Too high
        'pressure': 95.0,     # Too low
        'ph': 7.0            # Normal
    }
    
    anomalies = detector.detect_anomalies(anomalous_data)
    if anomalies:
        print("Anomalies detected:")
        for anomaly in anomalies:
            print(f"  • {anomaly['parameter']}: {anomaly['value']} "
                  f"(Z-score: {anomaly['z_score']:.2f}, Severity: {anomaly['severity']})")

if __name__ == "__main__":
    demo_predictive_models()