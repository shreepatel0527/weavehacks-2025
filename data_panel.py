import streamlit as st
import pandas as pd

def render_data_panel():
    """Render the side panel with experimental data tables"""
    st.header("Experimental Data")
    
    # Quantitative Observation Table
    st.subheader("Quantitative Observation")
    
    # Create a dataframe for quantitative data
    quant_data = {
        'Substance': [
            'HAuCl₄·3H₂O',
            'Water (for gold)',
            'TOAB',
            'Toluene',
            'PhCH₂CH₂SH',
            'NaBH₄',
            'Ice-cold Nanopure water (for NaBH₄)',
            'Final mass of nanoparticles'
        ],
        'Mass (g)': [''] * 8,
        'Volume (mL)': [''] * 8
    }
    
    df_quant = pd.DataFrame(quant_data)
    edited_df_quant = st.data_editor(df_quant, hide_index=True, key="quantitative_table")
    
    # Qualitative Observation Table
    st.subheader("Qualitative Observation")
    
    # Create a dataframe for qualitative data
    qual_data = {
        'Step': ['Step 4', 'Step 8', 'Step 25', 'Step 27'],
        'Color': [''] * 4,
        'Separation of Oil': [''] * 4,
        'Particle Formation': [''] * 4
    }
    
    df_qual = pd.DataFrame(qual_data)
    edited_df_qual = st.data_editor(df_qual, hide_index=True, key="qualitative_table")
    
    # Final volume measurement
    st.number_input("29) AU25 (mL)", min_value=0.0, step=0.1, key="au25_volume")


# Original analytics panel code (commented out)
'''
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

def create_sample_data():
    """Create sample data for demonstration"""
    if 'demo_data' not in st.session_state:
        # Create sample data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        data = {
            'Date': dates,
            'Messages': [i * 2 + 5 for i in range(30)],
            'Response Time': [i * 0.5 + 2 for i in range(30)]
        }
        st.session_state.demo_data = pd.DataFrame(data)

def render_data_panel():
    """Render the side panel with data visualization"""
    st.header("Analytics Panel")
    
    # Initialize demo data
    create_sample_data()
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Messages", len(st.session_state.messages))
    with col2:
        avg_response_time = 2.5  # Replace with actual calculation
        st.metric("Avg Response Time", f"{avg_response_time:.1f}s")
    
    # Message trend chart
    st.subheader("Message History")
    fig = px.line(st.session_state.demo_data, 
                 x='Date', 
                 y='Messages',
                 title='Daily Message Count')
    st.plotly_chart(fig, use_container_width=True)
    
    # Response time chart
    st.subheader("Response Times")
    fig = px.line(st.session_state.demo_data,
                 x='Date',
                 y='Response Time',
                 title='AI Response Times')
    st.plotly_chart(fig, use_container_width=True)
'''