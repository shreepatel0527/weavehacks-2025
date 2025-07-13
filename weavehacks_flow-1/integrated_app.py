def display_protocol_steps():
    """Protocol step management interface"""
    if not st.session_state.current_experiment_id:
        st.warning("Please create an experiment first in the Dashboard tab.")
        return
    
    # Import and use step panel
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend', 'components'))
        from step_panel import render_step_panel
        render_step_panel(st.session_state.current_experiment_id)
    except (ImportError, Exception) as e:
        # Fallback simple step panel
        render_simple_step_panel()
def display_data_panel():
    """Data panel interface"""
    if not st.session_state.current_experiment_id:
        st.warning("Please create an experiment first in the Dashboard tab.")
        return
    
    # Import and use data panel
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'frontend', 'components'))
        from data_panel import render_experiment_data_panel
        render_experiment_data_panel(st.session_state.current_experiment_id)
    except (ImportError, Exception) as e:
        # Fallback simple data panel
        render_simple_data_panel()
def render_simple_step_panel():
    """Simple fallback step panel"""
    st.subheader("📋 Protocol Steps")
    
    # Get protocol steps from API
    try:
        response = requests.get(f"{API_BASE_URL}/protocol/steps")
        if response.status_code == 200:
            data = response.json()
            steps = data["steps"]
        else:
            steps = []
    except:
        steps = []
    
    if not steps:
        st.error("Could not load protocol steps")
        return
    
    # Get current experiment step
    platform = st.session_state.platform
    success, exp_data = platform.get_experiment(st.session_state.current_experiment_id)
    current_step = exp_data.get('step_num', 0) if success else 0
    
    # Display current step
    if current_step < len(steps):
        step_info = steps[current_step]
        
        st.markdown(f"### Step {current_step + 1}: {step_info['title']}")
        st.markdown(f"**Description:** {step_info['description']}")
        st.markdown(f"**Details:** {step_info['details']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"⏱️ **Time:** {step_info['estimated_time']}")
        with col2:
            st.warning(f"⚠️ **Safety:** {step_info['safety_notes']}")
        
        # Navigation
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("⬅️ Previous", disabled=current_step <= 0):
                new_step = max(0, current_step - 1)
                update_experiment_step_api(st.session_state.current_experiment_id, new_step)
                st.rerun()
        
        with col2:
            st.metric("Progress", f"{current_step + 1}/{len(steps)}")
        
        with col3:
            if st.button("Next ➡️", disabled=current_step >= len(steps) - 1):
                new_step = min(len(steps) - 1, current_step + 1)
                update_experiment_step_api(st.session_state.current_experiment_id, new_step)
                st.rerun()
    
    # Step overview
    with st.expander("📋 All Steps Overview", expanded=False):
        for i, step in enumerate(steps):
            status = "✅" if i < current_step else "🟡" if i == current_step else "⚪"
            st.write(f"{status} **Step {i+1}:** {step['title']} ({step['estimated_time']})")
def render_simple_data_panel():
    """Simple fallback data panel"""
    st.subheader("📊 Experimental Data")
    
    # Get current experiment data
    platform = st.session_state.platform
    success, exp_data = platform.get_experiment(st.session_state.current_experiment_id)
    
    if not success:
        st.error("Could not load experiment data")
        return
    
    # Create simple data table
    data_items = [
        ("HAuCl₄·3H₂O", exp_data.get('mass_gold', 0), "g"),
        ("TOAB", exp_data.get('mass_toab', 0), "g"),
        ("PhCH₂CH₂SH", exp_data.get('mass_sulfur', 0), "g"),
        ("NaBH₄", exp_data.get('mass_nabh4', 0), "g"),
        ("Final Au₂₅", exp_data.get('mass_final', 0), "g"),
        ("Nanopure (RT)", exp_data.get('volume_nanopure_rt', 0), "mL"),
        ("Toluene", exp_data.get('volume_toluene', 0), "mL"),
        ("Nanopure (Cold)", exp_data.get('volume_nanopure_cold', 0), "mL")
    ]
    
    st.write("**Current Measurements:**")
    
    for substance, value, units in data_items:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(substance)
        with col2:
            if value > 0:
                st.success(f"{value:.4f}" if units == "g" else f"{value:.2f}")
            else:
                st.info("Not recorded")
        with col3:
            st.write(units)
    
    # Quick data entry
    st.subheader("🚀 Quick Entry")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.form("quick_mass_entry"):
            compound = st.selectbox("Compound", [
                "HAuCl₄·3H₂O", "TOAB", "PhCH₂CH₂SH", "NaBH₄", "Au₂₅ nanoparticles"
            ])
            mass_val = st.number_input("Mass (g)", min_value=0.0, step=0.0001)
            
            if st.form_submit_button("Record Mass"):
                data_type = "mass"
                success = record_measurement_simple(compound, mass_val, "g", data_type)
                if success:
                    st.success("Recorded!")
                    st.rerun()
    
    with col2:
        with st.form("quick_volume_entry"):
            liquid = st.selectbox("Liquid", [
                "nanopure water", "toluene", "ice-cold nanopure water"
            ])
            vol_val = st.number_input("Volume (mL)", min_value=0.0, step=0.01)
            
            if st.form_submit_button("Record Volume"):
                data_type = "volume"
                success = record_measurement_simple(liquid, vol_val, "mL", data_type)
                if success:
                    st.success("Recorded!")
                    st.rerun()
def update_experiment_step_api(experiment_id: str, step_num: int):
    """Update experiment step via API"""
    try:
        response = requests.put(f"{API_BASE_URL}/experiments/{experiment_id}/step",
                               json={"step_num": step_num})
        return response.status_code == 200
    except:
        return False
def record_measurement_simple(compound: str, value: float, units: str, data_type: str):
    """Simple measurement recording"""
    platform = st.session_state.platform
    return platform.record_data_via_api(
        st.session_state.current_experiment_id, data_type, compound, value, units
    )[0]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🤖 AI Assistant",
        "📋 Protocol Steps",
        "📈 Data Panel"
    with tab5:
        display_protocol_steps()
    
    with tab6:
        display_data_panel()
    
