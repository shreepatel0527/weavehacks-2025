#!/usr/bin/env python3
"""
WeaveHacks 2025 - Lab Automation Platform Backend
FastAPI backend for orchestrating lab experiments with AI agents
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import uvicorn
import logging
import asyncio
from datetime import datetime

# Try to import weave, but make it optional
try:
    import weave
    WEAVE_AVAILABLE = True
    weave.init('weavehacks-lab-backend')
    print("Weave initialized for backend tracking")
except Exception as e:
    WEAVE_AVAILABLE = False
    print(f"Weave not available (optional): {e}")

# Decorator for optional weave
def weave_op():
    def decorator(func):
        if WEAVE_AVAILABLE:
            return weave.op()(func)
        return func
    return decorator

# Try to import sensor simulator
try:
    from sensor_simulator import sensor_simulator, SensorType
    SENSOR_SIM_AVAILABLE = True
except ImportError:
    SENSOR_SIM_AVAILABLE = False
    print("Sensor simulator not available")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="WeaveHacks Lab Automation API",
    description="AI-powered lab automation platform for wet lab scientists",
    version="1.0.0"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class ExperimentState(BaseModel):
    experiment_id: str
    step_num: int = 0
    status: str = "not_started"  # not_started, in_progress, completed, error
    # Masses (g)
    mass_gold: float = 0.0
    mass_toab: float = 0.0
    mass_sulfur: float = 0.0
    mass_nabh4: float = 0.0
    mass_final: float = 0.0
    # Volumes (mL)
    volume_toluene: float = 0.0
    volume_nanopure_rt: float = 0.0
    volume_nanopure_cold: float = 0.0
    # Safety and observations
    safety_status: str = "safe"
    observations: str = ""
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()

class DataEntry(BaseModel):
    experiment_id: str
    data_type: str  # "mass" or "volume"
    compound: str
    value: float
    units: str
    timestamp: datetime = datetime.now()

class SafetyAlert(BaseModel):
    experiment_id: str
    parameter: str
    value: float
    threshold: float
    severity: str  # "warning", "critical"
    timestamp: datetime = datetime.now()

class AgentStatus(BaseModel):
    agent_id: str
    agent_type: str  # "data_collection", "lab_control", "safety_monitoring"
    status: str  # "active", "idle", "error"
    current_task: Optional[str] = None
    last_updated: datetime = datetime.now()

# In-memory storage (replace with database in production)
experiments: Dict[str, ExperimentState] = {}
data_entries: List[DataEntry] = []
safety_alerts: List[SafetyAlert] = []
agent_statuses: Dict[str, AgentStatus] = {}

# API Endpoints

@app.get("/")
async def root():
    return {"message": "WeaveHacks Lab Automation Platform API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

# Experiment Management
@app.post("/experiments", response_model=ExperimentState)
@weave_op()
async def create_experiment(experiment_id: str):
    """Create a new experiment"""
    if experiment_id in experiments:
        raise HTTPException(status_code=400, detail="Experiment already exists")
    
    experiment = ExperimentState(experiment_id=experiment_id)
    experiments[experiment_id] = experiment
    
    logger.info(f"Created experiment: {experiment_id}")
    return experiment

@app.get("/experiments/{experiment_id}", response_model=ExperimentState)
async def get_experiment(experiment_id: str):
    """Get experiment details"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return experiments[experiment_id]

@app.get("/experiments", response_model=List[ExperimentState])
async def list_experiments():
    """List all experiments"""
    return list(experiments.values())

@app.put("/experiments/{experiment_id}/status")
@weave_op()
async def update_experiment_status(experiment_id: str, status: str):
    """Update experiment status"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    experiments[experiment_id].status = status
    experiments[experiment_id].updated_at = datetime.now()
    
    logger.info(f"Updated experiment {experiment_id} status to {status}")
    return {"message": "Status updated", "status": status}

# Data Collection
@app.post("/data", response_model=DataEntry)
@weave_op()
async def record_data(data: DataEntry):
    """Record experimental data"""
    if data.experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    # Update experiment state
    exp = experiments[data.experiment_id]
    if data.data_type == "mass":
        if "gold" in data.compound.lower():
            exp.mass_gold = data.value
        elif "toab" in data.compound.lower():
            exp.mass_toab = data.value
        elif "sulfur" in data.compound.lower() or "phch" in data.compound.lower():
            exp.mass_sulfur = data.value
        elif "nabh4" in data.compound.lower():
            exp.mass_nabh4 = data.value
        elif "final" in data.compound.lower() or "au25" in data.compound.lower():
            exp.mass_final = data.value
    elif data.data_type == "volume":
        if "toluene" in data.compound.lower():
            exp.volume_toluene = data.value
        elif "cold" in data.compound.lower():
            exp.volume_nanopure_cold = data.value
        else:
            exp.volume_nanopure_rt = data.value
    
    exp.updated_at = datetime.now()
    data_entries.append(data)
    
    logger.info(f"Recorded {data.data_type} data for {data.compound}: {data.value} {data.units}")
    return data

@app.get("/data/{experiment_id}", response_model=List[DataEntry])
async def get_experiment_data(experiment_id: str):
    """Get all data for an experiment"""
    return [d for d in data_entries if d.experiment_id == experiment_id]

# Chemistry Calculations
@app.post("/calculations/sulfur_amount")
@weave_op()
async def calculate_sulfur_amount(experiment_id: str, gold_mass: float):
    """Calculate amount of sulfur compound needed"""
    # Constants
    MW_HAuCl4_3H2O = 393.83
    MW_PhCH2CH2SH = 138.23
    equivalents = 3
    
    moles_gold = gold_mass / MW_HAuCl4_3H2O
    moles_sulfur = moles_gold * equivalents
    mass_sulfur = moles_sulfur * MW_PhCH2CH2SH
    
    result = {
        "moles_gold": round(moles_gold, 6),
        "moles_sulfur": round(moles_sulfur, 6),
        "mass_sulfur_g": round(mass_sulfur, 4),
        "equivalents": equivalents
    }
    
    logger.info(f"Calculated sulfur amount for {experiment_id}: {mass_sulfur:.4f}g")
    return result

@app.post("/calculations/nabh4_amount")
@weave_op()
async def calculate_nabh4_amount(experiment_id: str, gold_mass: float):
    """Calculate amount of NaBH4 needed"""
    # Constants
    MW_HAuCl4_3H2O = 393.83
    MW_NaBH4 = 37.83
    equivalents = 10
    
    moles_gold = gold_mass / MW_HAuCl4_3H2O
    moles_nabh4 = moles_gold * equivalents
    mass_nabh4 = moles_nabh4 * MW_NaBH4
    
    result = {
        "moles_gold": round(moles_gold, 6),
        "moles_nabh4": round(moles_nabh4, 6),
        "mass_nabh4_g": round(mass_nabh4, 4),
        "equivalents": equivalents
    }
    
    logger.info(f"Calculated NaBH4 amount for {experiment_id}: {mass_nabh4:.4f}g")
    return result

@app.post("/calculations/percent_yield")
@weave_op()
async def calculate_percent_yield(experiment_id: str, gold_mass: float, actual_yield: float):
    """Calculate percent yield"""
    # Constants
    MW_HAuCl4_3H2O = 393.83
    MW_Au = 196.97
    
    # Calculate theoretical yield
    gold_fraction = MW_Au / MW_HAuCl4_3H2O
    theoretical_gold_mass = gold_mass * gold_fraction
    
    # For Au25 clusters, account for ligands (~30% by mass)
    ligand_fraction = 0.3
    theoretical_yield = theoretical_gold_mass / (1 - ligand_fraction)
    
    # Calculate percent yield
    percent_yield = (actual_yield / theoretical_yield) * 100 if theoretical_yield > 0 else 0
    
    result = {
        "starting_mass_g": round(gold_mass, 4),
        "gold_content_g": round(theoretical_gold_mass, 4),
        "theoretical_yield_g": round(theoretical_yield, 4),
        "actual_yield_g": round(actual_yield, 4),
        "percent_yield": round(percent_yield, 2)
    }
    
    logger.info(f"Calculated percent yield for {experiment_id}: {percent_yield:.2f}%")
    return result

# Safety Monitoring
@app.post("/safety/alert")
@weave_op()
async def record_safety_alert(alert: SafetyAlert):
    """Record a safety alert"""
    if alert.experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    safety_alerts.append(alert)
    
    # Update experiment safety status
    if alert.severity == "critical":
        experiments[alert.experiment_id].safety_status = "critical"
    elif alert.severity == "warning" and experiments[alert.experiment_id].safety_status == "safe":
        experiments[alert.experiment_id].safety_status = "warning"
    
    logger.warning(f"Safety alert for {alert.experiment_id}: {alert.parameter} = {alert.value} ({alert.severity})")
    return {"message": "Alert recorded", "alert_id": len(safety_alerts)}

@app.get("/safety/{experiment_id}/alerts", response_model=List[SafetyAlert])
async def get_safety_alerts(experiment_id: str):
    """Get safety alerts for an experiment"""
    return [a for a in safety_alerts if a.experiment_id == experiment_id]

# Sensor Simulation (if available)
if SENSOR_SIM_AVAILABLE:
    @app.post("/sensors/{experiment_id}/start")
    async def start_sensors(experiment_id: str, background_tasks: BackgroundTasks):
        """Start sensor monitoring for an experiment"""
        if experiment_id not in experiments:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        background_tasks.add_task(monitor_sensors, experiment_id)
        return {"message": "Sensor monitoring started"}

    async def monitor_sensors(experiment_id: str):
        """Background task to monitor sensors"""
        while experiment_id in experiments and experiments[experiment_id].status == "in_progress":
            readings = sensor_simulator.get_all_readings()
            
            for sensor_type, reading in readings.items():
                if reading.status == "critical":
                    alert = SafetyAlert(
                        experiment_id=experiment_id,
                        parameter=sensor_type.value,
                        value=reading.value,
                        threshold=reading.threshold,
                        severity="critical"
                    )
                    await record_safety_alert(alert)
            
            await asyncio.sleep(5)  # Check every 5 seconds

# Agent Status
@app.put("/agents/{agent_id}/status")
async def update_agent_status(agent_id: str, status: AgentStatus):
    """Update agent status"""
    agent_statuses[agent_id] = status
    logger.info(f"Updated agent {agent_id} status: {status.status}")
    return {"message": "Agent status updated"}

@app.get("/agents", response_model=List[AgentStatus])
async def get_all_agents():
    """Get status of all agents"""
    return list(agent_statuses.values())

# Voice Data Entry Support
@app.post("/voice/transcribe")
@weave_op()
async def transcribe_voice(experiment_id: str, audio_text: str):
    """Process transcribed voice input"""
    # Simple parsing logic - in production, use NLP
    lower_text = audio_text.lower()
    
    # Try to extract numerical value
    import re
    numbers = re.findall(r'\d+\.?\d*', audio_text)
    if not numbers:
        return {"success": False, "message": "No numerical value found"}
    
    value = float(numbers[0])
    
    # Determine what was measured
    if any(word in lower_text for word in ["gold", "haucl"]):
        data = DataEntry(
            experiment_id=experiment_id,
            data_type="mass",
            compound="HAuCl₄·3H₂O",
            value=value,
            units="g"
        )
    elif "toab" in lower_text:
        data = DataEntry(
            experiment_id=experiment_id,
            data_type="mass",
            compound="TOAB",
            value=value,
            units="g"
        )
    elif any(word in lower_text for word in ["water", "nanopure"]):
        data = DataEntry(
            experiment_id=experiment_id,
            data_type="volume",
            compound="nanopure water",
            value=value,
            units="mL"
        )
    elif "toluene" in lower_text:
        data = DataEntry(
            experiment_id=experiment_id,
            data_type="volume",
            compound="toluene",
            value=value,
            units="mL"
        )
    else:
        return {"success": False, "message": "Could not determine measurement type"}
    
    result = await record_data(data)
    return {"success": True, "data": result}

# Experiment Types (if sensor sim available)
if SENSOR_SIM_AVAILABLE:
    @app.get("/experiment-types")
    async def get_experiment_types():
        """Get available experiment types"""
        return {
            "types": {
                name: {
                    "description": f"{name.replace('_', ' ').title()} synthesis",
                    "temperature_range": profile.temp_range,
                    "pressure_range": profile.pressure_range,
                    "duration_hours": profile.duration_hours
                }
                for name, profile in sensor_simulator.experiment_profiles.items()
            }
        }

# Step Management Endpoints (NEW)
@app.put("/experiments/{experiment_id}/step")
@weave_op()
async def update_experiment_step(experiment_id: str, step_data: dict):
    """Update experiment step number"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    step_num = step_data.get("step_num", 0)
    experiments[experiment_id].step_num = step_num
    experiments[experiment_id].updated_at = datetime.now()
    
    logger.info(f"Updated experiment {experiment_id} to step {step_num}")
    return {"message": "Step updated", "step_num": step_num}

@app.post("/experiments/{experiment_id}/steps/complete")
@weave_op()
async def complete_experiment_step(experiment_id: str, step_info: dict):
    """Mark a step as completed"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    step_id = step_info.get("step_id")
    step_title = step_info.get("step_title", f"Step {step_id}")
    
    # Store completion info (in production, use proper database)
    completion_data = {
        "step_id": step_id,
        "step_title": step_title,
        "completed_at": datetime.now().isoformat(),
        "experiment_id": experiment_id
    }
    
    logger.info(f"Step {step_id} completed for experiment {experiment_id}")
    return {"message": "Step marked as complete", "completion": completion_data}

@app.post("/experiments/{experiment_id}/steps/note")
@weave_op()
async def add_step_note(experiment_id: str, note_data: dict):
    """Add a note to a specific step"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    step_id = note_data.get("step_id")
    note = note_data.get("note", "")
    
    # Store note (in production, use proper database)
    note_entry = {
        "step_id": step_id,
        "note": note,
        "created_at": datetime.now().isoformat(),
        "experiment_id": experiment_id
    }
    
    logger.info(f"Note added to step {step_id} for experiment {experiment_id}")
    return {"message": "Note saved", "note": note_entry}

@app.put("/experiments/{experiment_id}/observations")
@weave_op()
async def update_experiment_observations(experiment_id: str, obs_data: dict):
    """Update experiment qualitative observations"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    observations = obs_data.get("observations", "")
    experiments[experiment_id].observations = observations
    experiments[experiment_id].updated_at = datetime.now()
    
    logger.info(f"Updated observations for experiment {experiment_id}")
    return {"message": "Observations updated", "observations": observations}

# Protocol Management Endpoints
@app.get("/protocol/steps")
async def get_protocol_steps():
    """Get the complete nanoparticle synthesis protocol"""
    steps = [
        {
            "id": 0,
            "title": "Setup and Preparation",
            "description": "Gather all reagents and equipment",
            "details": "Collect HAuCl₄·3H₂O, TOAB, nanopure water, toluene, 25 mL tri-neck round-bottom flask",
            "estimated_time": "5 min",
            "safety_notes": "Ensure fume hood is operational and wear appropriate PPE",
            "required_data": []
        },
        {
            "id": 1,
            "title": "Weigh HAuCl₄·3H₂O",
            "description": "Weigh 0.1576g HAuCl₄·3H₂O",
            "details": "Use analytical balance to accurately weigh the gold compound. Record exact mass.",
            "estimated_time": "3 min",
            "safety_notes": "Avoid inhalation of gold compound dust",
            "required_data": ["mass_gold"]
        },
        {
            "id": 2,
            "title": "Measure Nanopure Water (RT)",
            "description": "Measure 5 mL nanopure water at room temperature",
            "details": "Use graduated cylinder or volumetric pipette for accurate measurement",
            "estimated_time": "2 min",
            "safety_notes": "Ensure water is at room temperature",
            "required_data": ["volume_nanopure_rt"]
        },
        {
            "id": 3,
            "title": "Dissolve Gold Compound",
            "description": "Dissolve HAuCl₄·3H₂O in nanopure water",
            "details": "Add gold compound to water slowly with stirring until fully dissolved",
            "estimated_time": "5 min",
            "safety_notes": "Stir gently to avoid splashing",
            "required_data": [],
            "qualitative_check": "Solution should be clear yellow/orange"
        },
        {
            "id": 4,
            "title": "Weigh TOAB",
            "description": "Weigh approximately 0.25g TOAB",
            "details": "Weigh tetraoctylammonium bromide on analytical balance",
            "estimated_time": "3 min",
            "safety_notes": "TOAB is hygroscopic, minimize exposure to air",
            "required_data": ["mass_toab"]
        },
        {
            "id": 5,
            "title": "Measure Toluene",
            "description": "Measure 10 mL toluene",
            "details": "Use graduated cylinder to measure organic solvent",
            "estimated_time": "2 min",
            "safety_notes": "Work in fume hood - toluene vapors are toxic",
            "required_data": ["volume_toluene"]
        },
        {
            "id": 6,
            "title": "Dissolve TOAB in Toluene",
            "description": "Dissolve TOAB in toluene",
            "details": "Add TOAB to toluene and stir until completely dissolved",
            "estimated_time": "5 min",
            "safety_notes": "Ensure good ventilation",
            "required_data": [],
            "qualitative_check": "Solution should be clear and colorless"
        },
        {
            "id": 7,
            "title": "Combine Solutions",
            "description": "Combine aqueous and organic phases in round-bottom flask",
            "details": "Add both solutions to 25 mL tri-neck round-bottom flask",
            "estimated_time": "3 min",
            "safety_notes": "Handle flask carefully to avoid breakage",
            "required_data": [],
            "qualitative_check": "Two distinct phases should be visible"
        },
        {
            "id": 8,
            "title": "Transfer to Fume Hood",
            "description": "Move setup to fume hood",
            "details": "Carefully transfer apparatus to fume hood for vigorous stirring",
            "estimated_time": "2 min",
            "safety_notes": "Secure all connections before moving",
            "required_data": []
        },
        {
            "id": 9,
            "title": "Vigorous Stirring",
            "description": "Stir vigorously at ~1100 rpm for 15 minutes",
            "details": "Place on stir plate with stir bar, maintain high stirring speed",
            "estimated_time": "15 min",
            "safety_notes": "Monitor for proper mixing and phase transfer",
            "required_data": [],
            "qualitative_check": "Good emulsion formation, gold transfer to organic phase"
        },
        {
            "id": 10,
            "title": "Remove Aqueous Layer",
            "description": "Remove aqueous layer with 10 mL syringe",
            "details": "Carefully extract the lower aqueous phase",
            "estimated_time": "5 min",
            "safety_notes": "Avoid disturbing organic phase",
            "required_data": []
        },
        {
            "id": 11,
            "title": "Setup for Reduction",
            "description": "Place gasket and gas needle, purge with N₂, cool to 0°C",
            "details": "Seal flask, establish nitrogen atmosphere, place in ice bath",
            "estimated_time": "30 min",
            "safety_notes": "Ensure proper N₂ flow and ice bath stability",
            "required_data": []
        }
    ]
    
    return {"steps": steps, "total_steps": len(steps)}

# Data Export Endpoints
@app.get("/experiments/{experiment_id}/export/csv")
async def export_experiment_csv(experiment_id: str):
    """Export experiment data as CSV"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    exp = experiments[experiment_id]
    
    csv_content = f"""Experiment ID,{experiment_id}
Created,{exp.created_at}
Status,{exp.status}
Step,{exp.step_num}

Substance,Mass (g),Volume (mL)
HAuCl₄·3H₂O,{exp.mass_gold},
TOAB,{exp.mass_toab},
PhCH₂CH₂SH,{exp.mass_sulfur},
NaBH₄,{exp.mass_nabh4},
Final Au₂₅,{exp.mass_final},
Nanopure (RT),,{exp.volume_nanopure_rt}
Toluene,,{exp.volume_toluene}
Nanopure (Cold),,{exp.volume_nanopure_cold}

Observations
{exp.observations}
"""
    
    return {"csv_content": csv_content, "filename": f"experiment_{experiment_id}.csv"}

@app.get("/experiments/{experiment_id}/export/report")
async def export_experiment_report(experiment_id: str):
    """Export detailed experiment report"""
    if experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    exp = experiments[experiment_id]
    
    # Calculate percent yield if possible
    percent_yield = "N/A"
    if exp.mass_gold > 0 and exp.mass_final > 0:
        try:
            result = await calculate_percent_yield(experiment_id, exp.mass_gold, exp.mass_final)
            percent_yield = f"{result['percent_yield']:.2f}%"
        except:
            percent_yield = "Calculation Error"
    
    report_content = f"""
NANOPARTICLE SYNTHESIS EXPERIMENT REPORT
======================================

Experiment Information:
- ID: {experiment_id}
- Created: {exp.created_at}
- Last Updated: {exp.updated_at}
- Status: {exp.status.title()}
- Current Step: {exp.step_num}/12
- Safety Status: {exp.safety_status.title()}

Quantitative Data:
- HAuCl₄·3H₂O: {exp.mass_gold:.4f} g
- TOAB: {exp.mass_toab:.4f} g
- PhCH₂CH₂SH: {exp.mass_sulfur:.4f} g
- NaBH₄: {exp.mass_nabh4:.4f} g
- Final Au₂₅: {exp.mass_final:.4f} g

- Nanopure water (RT): {exp.volume_nanopure_rt:.2f} mL
- Toluene: {exp.volume_toluene:.2f} mL
- Nanopure water (Cold): {exp.volume_nanopure_cold:.2f} mL

Analysis:
- Percent Yield: {percent_yield}

Qualitative Observations:
{exp.observations if exp.observations else 'No observations recorded'}

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return {"report_content": report_content, "filename": f"report_{experiment_id}.txt"}

# WebSocket endpoint for real-time updates (placeholder)
@app.get("/ws")
async def websocket_info():
    return {"message": "WebSocket endpoint for real-time updates", "endpoint": "/ws"}

if __name__ == "__main__":
    # Start sensor simulation by default if available
    if SENSOR_SIM_AVAILABLE:
        sensor_simulator.start_simulation()
    
    uvicorn.run(app, host="0.0.0.0", port=8000)