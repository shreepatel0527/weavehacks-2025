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
