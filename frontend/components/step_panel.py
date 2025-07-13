#!/usr/bin/env python3
"""
WeaveHacks 2025 - Enhanced Step Panel  
Interactive protocol step management for lab experiments
"""

import streamlit as st
import requests
from datetime import datetime
import time

API_BASE_URL = "http://localhost:8000"

def initialize_step_state():
    """Initialize step tracking state"""
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    
    if 'step_start_time' not in st.session_state:
        st.session_state.step_start_time = datetime.now()
    
    if 'step_notes' not in st.session_state:
        st.session_state.step_notes = {}
    
    if 'steps_completed' not in st.session_state:
        st.session_state.steps_completed = set()

def get_protocol_steps():
    """Get the complete Au nanoparticle synthesis protocol"""
    return [
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

def render_step_panel(experiment_id: str = None):
    """Render the enhanced step panel"""
    st.header("📋 Protocol Step Manager")
    
    initialize_step_state()
    steps = get_protocol_steps()
    
    if not experiment_id:
        st.warning("No active experiment. Please create an experiment first.")
        return
    
    # Get experiment data to check completion status
    try:
        response = requests.get(f"{API_BASE_URL}/experiments/{experiment_id}")
        if response.status_code == 200:
            exp_data = response.json()
            current_exp_step = exp_data.get('step_num', 0)
        else:
            current_exp_step = 0
    except:
        current_exp_step = 0
    
    # Sync with experiment step if different
    if current_exp_step != st.session_state.current_step:
        st.session_state.current_step = current_exp_step
    
    # Progress overview
    progress = (st.session_state.current_step / len(steps)) * 100
    st.progress(progress / 100, f"Protocol Progress: {progress:.1f}% (Step {st.session_state.current_step + 1}/{len(steps)})")
    
    # Current step display
    current_step_info = steps[min(st.session_state.current_step, len(steps) - 1)]
    
    # Step navigation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⬅️ Previous", disabled=st.session_state.current_step <= 0):
            if st.session_state.current_step > 0:
                st.session_state.current_step -= 1
                update_experiment_step(experiment_id, st.session_state.current_step)
                st.rerun()
    
    with col2:
        st.markdown(f"### Step {current_step_info['id'] + 1}: {current_step_info['title']}")
    
    with col3:
        if st.button("Next ➡️", disabled=st.session_state.current_step >= len(steps) - 1):
            if st.session_state.current_step < len(steps) - 1:
                st.session_state.current_step += 1
                update_experiment_step(experiment_id, st.session_state.current_step)
                st.rerun()
    
    # Current step details
    step_container = st.container()
    
    with step_container:
        # Step description and details
        st.markdown(f"**Description:** {current_step_info['description']}")
        st.markdown(f"**Details:** {current_step_info['details']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"⏱️ **Estimated Time:** {current_step_info['estimated_time']}")
        with col2:
            st.warning(f"⚠️ **Safety:** {current_step_info['safety_notes']}")
        
        # Data entry requirements
        if current_step_info.get('required_data'):
            st.subheader("📊 Required Data Entry")
            
            for data_field in current_step_info['required_data']:
                render_data_entry_field(data_field, experiment_id)
        
        # Qualitative check
        if current_step_info.get('qualitative_check'):
            st.subheader("🔍 Qualitative Check")
            st.success(f"✅ Expected Result: {current_step_info['qualitative_check']}")
            
            # Observation input
            obs_key = f"step_{current_step_info['id']}_observation"
            observation = st.text_area(
                "Record your observation:",
                value=st.session_state.step_notes.get(obs_key, ""),
                height=80,
                key=f"obs_{current_step_info['id']}"
            )
            st.session_state.step_notes[obs_key] = observation
        
        # Step completion
        step_completed = current_step_info['id'] in st.session_state.steps_completed
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("✅ Mark Complete", disabled=step_completed, key=f"complete_{current_step_info['id']}"):
                st.session_state.steps_completed.add(current_step_info['id'])
                save_step_completion(experiment_id, current_step_info['id'], current_step_info['title'])
                st.success(f"Step {current_step_info['id'] + 1} marked as complete!")
        
        with col2:
            if st.button("📝 Add Note", key=f"note_{current_step_info['id']}"):
                note_key = f"step_{current_step_info['id']}_note"
                if note_key not in st.session_state.step_notes:
                    st.session_state.step_notes[note_key] = ""
                st.session_state.show_note_input = current_step_info['id']
        
        with col3:
            if step_completed:
                st.success("✅ Completed")
            else:
                st.info("⏳ In Progress")
        
        # Note input (if requested)
        if getattr(st.session_state, 'show_note_input', None) == current_step_info['id']:
            note_key = f"step_{current_step_info['id']}_note"
            note = st.text_area(
                "Add a note for this step:",
                value=st.session_state.step_notes.get(note_key, ""),
                height=100,
                key=f"note_input_{current_step_info['id']}"
            )
            
            if st.button("Save Note", key=f"save_note_{current_step_info['id']}"):
                st.session_state.step_notes[note_key] = note
                save_step_note(experiment_id, current_step_info['id'], note)
                st.session_state.show_note_input = None
                st.success("Note saved!")
                st.rerun()
    
    # Step overview sidebar
    with st.sidebar:
        st.subheader("📋 Protocol Overview")
        
        for i, step in enumerate(steps):
            status_icon = "✅" if i in st.session_state.steps_completed else "⏳" if i == st.session_state.current_step else "⚪"
            step_status = "completed" if i in st.session_state.steps_completed else "current" if i == st.session_state.current_step else "pending"
            
            # Create clickable step
            if st.button(f"{status_icon} {i+1}. {step['title']}", 
                        key=f"sidebar_step_{i}",
                        help=step['description'],
                        use_container_width=True):
                st.session_state.current_step = i
                update_experiment_step(experiment_id, i)
                st.rerun()
        
        # Step statistics
        st.divider()
        completed_count = len(st.session_state.steps_completed)
        st.metric("Completed Steps", f"{completed_count}/{len(steps)}")
        
        # Time tracking
        if 'step_start_time' in st.session_state:
            elapsed = datetime.now() - st.session_state.step_start_time
            st.metric("Current Step Time", f"{elapsed.seconds // 60}:{elapsed.seconds % 60:02d}")

def render_data_entry_field(data_field: str, experiment_id: str):
    """Render data entry field for required step data"""
    field_mapping = {
        'mass_gold': ('HAuCl₄·3H₂O Mass', 'g', 'mass', 0.0001),
        'mass_toab': ('TOAB Mass', 'g', 'mass', 0.0001),
        'mass_sulfur': ('PhCH₂CH₂SH Mass', 'g', 'mass', 0.0001),
        'mass_nabh4': ('NaBH₄ Mass', 'g', 'mass', 0.0001),
        'mass_final': ('Final Au₂₅ Mass', 'g', 'mass', 0.0001),
        'volume_nanopure_rt': ('Nanopure Water (RT)', 'mL', 'volume', 0.01),
        'volume_toluene': ('Toluene Volume', 'mL', 'volume', 0.01),
        'volume_nanopure_cold': ('Cold Nanopure Water', 'mL', 'volume', 0.01)
    }
    
    if data_field in field_mapping:
        label, units, data_type, step = field_mapping[data_field]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            value = st.number_input(
                f"{label} ({units})",
                min_value=0.0,
                step=step,
                format=f"%.{len(str(step).split('.')[-1])}f" if '.' in str(step) else "%.0f",
                key=f"input_{data_field}"
            )
        
        with col2:
            if st.button("Record", key=f"record_{data_field}"):
                if value > 0:
                    success = record_measurement_api(experiment_id, label, value, units, data_type)
                    if success:
                        st.success(f"Recorded: {value} {units}")
                        st.rerun()
                    else:
                        st.error("Failed to record data")
                else:
                    st.warning("Please enter a value > 0")

def record_measurement_api(experiment_id: str, compound: str, value: float, units: str, data_type: str):
    """Record measurement via API"""
    data = {
        "experiment_id": experiment_id,
        "data_type": data_type,
        "compound": compound,
        "value": value,
        "units": units
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/data", json=data)
        return response.status_code == 200
    except:
        return False

def update_experiment_step(experiment_id: str, step_num: int):
    """Update experiment step via API"""
    try:
        response = requests.put(f"{API_BASE_URL}/experiments/{experiment_id}/step", 
                               json={"step_num": step_num})
        return response.status_code == 200
    except:
        return False

def save_step_completion(experiment_id: str, step_id: int, step_title: str):
    """Save step completion via API"""
    try:
        response = requests.post(f"{API_BASE_URL}/experiments/{experiment_id}/steps/complete",
                               json={"step_id": step_id, "step_title": step_title})
        return response.status_code == 200
    except:
        return False

def save_step_note(experiment_id: str, step_id: int, note: str):
    """Save step note via API"""
    try:
        response = requests.post(f"{API_BASE_URL}/experiments/{experiment_id}/steps/note",
                               json={"step_id": step_id, "note": note})
        return response.status_code == 200
    except:
        return False

def render_protocol_summary():
    """Render protocol summary view"""
    st.subheader("📋 Complete Protocol Summary")
    
    steps = get_protocol_steps()
    
    for i, step in enumerate(steps):
        with st.expander(f"Step {i+1}: {step['title']}", expanded=False):
            st.markdown(f"**Description:** {step['description']}")
            st.markdown(f"**Details:** {step['details']}")
            st.markdown(f"**Estimated Time:** {step['estimated_time']}")
            st.markdown(f"**Safety Notes:** {step['safety_notes']}")
            
            if step.get('required_data'):
                st.markdown(f"**Required Data:** {', '.join(step['required_data'])}")
            
            if step.get('qualitative_check'):
                st.markdown(f"**Expected Result:** {step['qualitative_check']}")

# Additional step panel features
def render_step_timer():
    """Render step timer functionality"""
    if 'step_timer_start' not in st.session_state:
        st.session_state.step_timer_start = None
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⏱️ Start Timer"):
            st.session_state.step_timer_start = datetime.now()
    
    with col2:
        if st.button("⏹️ Stop Timer"):
            if st.session_state.step_timer_start:
                elapsed = datetime.now() - st.session_state.step_timer_start
                st.success(f"Step completed in {elapsed.seconds // 60}:{elapsed.seconds % 60:02d}")
                st.session_state.step_timer_start = None
    
    with col3:
        if st.session_state.step_timer_start:
            elapsed = datetime.now() - st.session_state.step_timer_start
            st.info(f"Timer: {elapsed.seconds // 60}:{elapsed.seconds % 60:02d}")

def render_step_checklist(step_info: dict):
    """Render checklist for complex steps"""
    if step_info['id'] == 9:  # Vigorous stirring step
        st.subheader("✅ Step Checklist")
        
        checklist_items = [
            "Stir bar is properly placed and spinning",
            "Stirring speed is approximately 1100 rpm",
            "Good emulsion formation is observed",
            "Gold transfer to organic phase is visible",
            "Timer set for 15 minutes"
        ]
        
        for i, item in enumerate(checklist_items):
            st.checkbox(item, key=f"checklist_{step_info['id']}_{i}")
    
    elif step_info['id'] == 11:  # Setup for reduction
        st.subheader("✅ Setup Checklist")
        
        checklist_items = [
            "Gasket properly sealed",
            "Gas needle connected",
            "Nitrogen flow established",
            "Ice bath prepared and stable at 0°C",
            "Stirring maintained during cooling"
        ]
        
        for i, item in enumerate(checklist_items):
            st.checkbox(item, key=f"checklist_{step_info['id']}_{i}")