import streamlit as st

def initialize_step_state():
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'steps' not in st.session_state:
        st.session_state.steps = [
            "STEP 1: Grab the beaker",
            "STEP 2: Add HAuCl4·3H2O",
            "STEP 3: Add water",
            "STEP 4: Check solution quality",
            "STEP 5: Add TOAB"
            # Add more steps as needed
        ]

def next_step():
    if st.session_state.current_step < len(st.session_state.steps) - 1:
        st.session_state.current_step += 1

def render_step_panel():
    initialize_step_state()
    
    # Create a container for the step panel
    step_container = st.container()
    
    with step_container:
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.markdown(f"### {st.session_state.steps[st.session_state.current_step]}")
        
        with col2:
            if st.button("Next", key="next_step"):
                next_step()