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
import weave
from sensor_simulator import sensor_simulator, SensorType

# Initialize Weave for observability
try:
    weave.init('weavehacks-lab-backend')
    print("Weave initialized for backend tracking")
except Exception as e:
    print(f"Weave init failed (optional): {e}")

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
@weave.op()
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
@weave.op()
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
@weave.op()
async def record_data(data: DataEntry):
    """Record experimental data"""
    if data.experiment_id not in experiments:
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    data_entries.append(data)
    
    # Update experiment state based on data type
    experiment = experiments[data.experiment_id]
    experiment.updated_at = datetime.now()
    
    # Map data to experiment fields
    if data.compound.lower().contains("gold") or "haucl4" in data.compound.lower():
        experiment.mass_gold = data.value
    elif data.compound.lower().contains("toab"):
        experiment.mass_toab = data.value
    elif data.compound.lower().contains("sulfur") or "phch2ch2sh" in data.compound.lower():
        experiment.mass_sulfur = data.value
    elif data.compound.lower().contains("nabh4"):
        experiment.mass_nabh4 = data.value
    elif data.compound.lower().contains("final") or "nanoparticle" in data.compound.lower():
        experiment.mass_final = data.value
    elif data.compound.lower().contains("toluene"):
        experiment.volume_toluene = data.value
    elif "nanopure" in data.compound.lower() and "cold" in data.compound.lower():
        experiment.volume_nanopure_cold = data.value
    elif "nanopure" in data.compound.lower():
        experiment.volume_nanopure_rt = data.value
    
    logger.info(f"Recorded data for {data.experiment_id}: {data.compound} = {data.value} {data.units}")
    return data

@app.get("/experiments/{experiment_id}/data", response_model=List[DataEntry])
async def get_experiment_data(experiment_id: str):
    """Get all data for an experiment"""
    return [entry for entry in data_entries if entry.experiment_id == experiment_id]

# Safety Monitoring
@app.post("/safety/alert", response_model=SafetyAlert)
@weave.op()
async def create_safety_alert(alert: SafetyAlert):
    """Create a safety alert"""
    safety_alerts.append(alert)
    
    # Update experiment safety status if critical
    if alert.severity == "critical" and alert.experiment_id in experiments:
        experiments[alert.experiment_id].safety_status = "unsafe"
        experiments[alert.experiment_id].updated_at = datetime.now()
    
    logger.warning(f"Safety alert for {alert.experiment_id}: {alert.parameter} = {alert.value} (threshold: {alert.threshold})")
    return alert

@app.get("/safety/alerts", response_model=List[SafetyAlert])
async def get_safety_alerts():
    """Get all safety alerts"""
    return safety_alerts

@app.get("/experiments/{experiment_id}/safety", response_model=List[SafetyAlert])
async def get_experiment_safety_alerts(experiment_id: str):
    """Get safety alerts for an experiment"""
    return [alert for alert in safety_alerts if alert.experiment_id == experiment_id]

# Agent Management
@app.post("/agents", response_model=AgentStatus)
@weave.op()
async def register_agent(agent: AgentStatus):
    """Register an agent"""
    agent_statuses[agent.agent_id] = agent
    logger.info(f"Registered agent: {agent.agent_id} ({agent.agent_type})")
    return agent

@app.get("/agents", response_model=List[AgentStatus])
async def get_agents():
    """Get all agent statuses"""
    return list(agent_statuses.values())

@app.put("/agents/{agent_id}/status")
@weave.op()
async def update_agent_status(agent_id: str, status: str, current_task: Optional[str] = None):
    """Update agent status"""
    if agent_id not in agent_statuses:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    agent_statuses[agent_id].status = status
    if current_task:
        agent_statuses[agent_id].current_task = current_task
    agent_statuses[agent_id].last_updated = datetime.now()
    
    return {"message": "Agent status updated", "agent_id": agent_id, "status": status}

# Chemistry Calculations
@app.post("/calculations/sulfur-amount")
@weave.op()
async def calculate_sulfur_amount(experiment_id: str, gold_mass: float):
    """Calculate required sulfur amount (3 eq. relative to gold)"""
    # HAuCl4·3H2O molecular weight: 393.83 g/mol
    # PhCH2CH2SH molecular weight: 138.21 g/mol
    
    mw_haucl4_3h2o = 393.83
    mw_sulfur = 138.21
    
    moles_gold = gold_mass / mw_haucl4_3h2o
    moles_sulfur_needed = moles_gold * 3  # 3 equivalents
    mass_sulfur_needed = moles_sulfur_needed * mw_sulfur
    
    result = {
        "experiment_id": experiment_id,
        "gold_mass_g": gold_mass,
        "moles_gold": moles_gold,
        "moles_sulfur_needed": moles_sulfur_needed,
        "mass_sulfur_needed_g": mass_sulfur_needed,
        "calculation_type": "sulfur_amount"
    }
    
    logger.info(f"Calculated sulfur amount for {experiment_id}: {mass_sulfur_needed:.4f}g")
    return result

@app.post("/calculations/nabh4-amount")
@weave.op()
async def calculate_nabh4_amount(experiment_id: str, gold_mass: float):
    """Calculate required NaBH4 amount (10 eq. relative to gold)"""
    # HAuCl4·3H2O molecular weight: 393.83 g/mol
    # NaBH4 molecular weight: 37.83 g/mol
    
    mw_haucl4_3h2o = 393.83
    mw_nabh4 = 37.83
    
    moles_gold = gold_mass / mw_haucl4_3h2o
    moles_nabh4_needed = moles_gold * 10  # 10 equivalents
    mass_nabh4_needed = moles_nabh4_needed * mw_nabh4
    
    result = {
        "experiment_id": experiment_id,
        "gold_mass_g": gold_mass,
        "moles_gold": moles_gold,
        "moles_nabh4_needed": moles_nabh4_needed,
        "mass_nabh4_needed_g": mass_nabh4_needed,
        "calculation_type": "nabh4_amount"
    }
    
    logger.info(f"Calculated NaBH4 amount for {experiment_id}: {mass_nabh4_needed:.4f}g")
    return result

@app.post("/calculations/percent-yield")
@weave.op()
async def calculate_percent_yield(experiment_id: str, gold_mass: float, final_mass: float):
    """Calculate percent yield of the experiment"""
    # Assume theoretical yield equals gold content in starting material
    # HAuCl4·3H2O contains Au (atomic weight 196.97)
    # HAuCl4·3H2O molecular weight: 393.83 g/mol
    
    mw_haucl4_3h2o = 393.83
    mw_gold = 196.97
    
    moles_starting = gold_mass / mw_haucl4_3h2o
    theoretical_gold_mass = moles_starting * mw_gold
    
    percent_yield = (final_mass / theoretical_gold_mass) * 100 if theoretical_gold_mass > 0 else 0
    
    result = {
        "experiment_id": experiment_id,
        "starting_mass_g": gold_mass,
        "theoretical_gold_mass_g": theoretical_gold_mass,
        "actual_yield_g": final_mass,
        "percent_yield": percent_yield,
        "calculation_type": "percent_yield"
    }
    
    logger.info(f"Calculated percent yield for {experiment_id}: {percent_yield:.2f}%")
    return result

# Sensor Data Endpoints
@app.post("/sensors/start-experiment")
@weave.op()
async def start_sensor_experiment(experiment_type: str, experiment_id: str):
    """Start sensor simulation for an experiment"""
    success, message = sensor_simulator.start_experiment(experiment_type, experiment_id)
    
    if success:
        logger.info(f"Started sensor simulation for {experiment_id}: {experiment_type}")
        return {"success": True, "message": message}
    else:
        logger.error(f"Failed to start sensor simulation: {message}")
        raise HTTPException(status_code=400, detail=message)

@app.get("/sensors/readings")
async def get_sensor_readings(count: int = 50):
    """Get recent sensor readings"""
    readings = sensor_simulator.get_recent_readings(count)
    return {"readings": readings, "count": len(readings)}

@app.get("/sensors/status")
async def get_sensor_status():
    """Get sensor system status"""
    experiment_status = sensor_simulator.get_experiment_status()
    recent_readings = sensor_simulator.get_recent_readings(10)
    
    # Check for safety alerts
    from sensor_simulator import SensorReading, SensorType
    reading_objects = []
    for r in recent_readings:
        reading_objects.append(SensorReading(
            sensor_id=r["sensor_id"],
            sensor_type=SensorType(r["sensor_type"]),
            value=r["value"],
            units=r["units"],
            timestamp=datetime.fromisoformat(r["timestamp"]),
            location=r["location"],
            experiment_id=r.get("experiment_id")
        ))
    
    safety_alerts = sensor_simulator.check_safety_thresholds(reading_objects)
    
    return {
        "experiment": experiment_status,
        "recent_readings": recent_readings,
        "safety_alerts": safety_alerts,
        "sensor_count": len(sensor_simulator.sensors),
        "simulation_active": sensor_simulator.is_running
    }

@app.post("/sensors/stop")
@weave.op()
async def stop_sensor_simulation():
    """Stop sensor simulation"""
    sensor_simulator.stop_simulation()
    logger.info("Stopped sensor simulation")
    return {"message": "Sensor simulation stopped"}

@app.get("/sensors/experiment-types")
async def get_experiment_types():
    """Get available experiment types for sensor simulation"""
    return {
        "experiment_types": list(sensor_simulator.experiment_profiles.keys()),
        "profiles": {
            name: {
                "name": profile.name,
                "temperature_range": profile.temp_range,
                "pressure_range": profile.pressure_range,
                "duration_hours": profile.duration_hours
            }
            for name, profile in sensor_simulator.experiment_profiles.items()
        }
    }

# Step Management Endpoints
@app.put("/experiments/{experiment_id}/step")
@weave.op()
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
@weave.op()
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
@weave.op()
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
@weave.op()
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
            success, result = await calculate_percent_yield(experiment_id, exp.mass_gold, exp.mass_final)
            if success:
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
    # Start sensor simulation by default
    sensor_simulator.start_simulation()
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)