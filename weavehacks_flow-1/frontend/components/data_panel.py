#!/usr/bin/env python3
"""
WeaveHacks 2025 - Enhanced Data Panel
Comprehensive data entry and visualization for lab experiments
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import requests
API_BASE_URL = "http://localhost:8000"
def render_experiment_data_panel(experiment_id: str = None):
    """Render comprehensive experimental data panel"""
    st.header("🧪 Experimental Data Panel")
    
    if not experiment_id:
        st.warning("No active experiment. Please create an experiment first.")
        return
    
    # Get current experiment data
    try:
        response = requests.get(f"{API_BASE_URL}/experiments/{experiment_id}")
        if response.status_code == 200:
            exp_data = response.json()
        else:
            st.error("Failed to load experiment data")
            return
    except:
        st.error("Cannot connect to backend API")
        return
    
    # Create tabs for different data views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Quantitative Data", 
        "🔍 Qualitative Observations", 
        "📈 Data Visualization",
        "📋 Data Export"
    ])
    
    with tab1:
        render_quantitative_panel(exp_data, experiment_id)
    
    with tab2:
        render_qualitative_panel(experiment_id)
    
    with tab3:
        render_visualization_panel(exp_data)
    
    with tab4:
        render_export_panel(exp_data, experiment_id)
def render_quantitative_panel(exp_data: dict, experiment_id: str):
    """Render quantitative data entry and display"""
    st.subheader("⚖️ Quantitative Measurements")
    
    # Create enhanced quantitative data table
    substances = [
        'HAuCl₄·3H₂O',
        'Water (for gold)',
        'TOAB', 
        'Toluene',
        'PhCH₂CH₂SH',
        'NaBH₄',
        'Ice-cold Nanopure water (for NaBH₄)',
        'Final mass of Au₂₅ nanoparticles'
    ]
    
    # Get current masses and volumes from experiment data
    current_masses = [
        exp_data.get('mass_gold', 0.0),
        0.0,  # Water mass (calculated from volume)
        exp_data.get('mass_toab', 0.0),
        0.0,  # Toluene mass (calculated from volume)
        exp_data.get('mass_sulfur', 0.0),
        exp_data.get('mass_nabh4', 0.0),
        0.0,  # Cold water mass
        exp_data.get('mass_final', 0.0)
    ]
    
    current_volumes = [
        0.0,  # Gold compound (solid)
        exp_data.get('volume_nanopure_rt', 0.0),
        0.0,  # TOAB (solid)
        exp_data.get('volume_toluene', 0.0),
        0.0,  # Sulfur compound (liquid, but measured by mass)
        0.0,  # NaBH4 (solid)
        exp_data.get('volume_nanopure_cold', 0.0),
        0.0   # Final product (solid)
    ]
    
    # Create DataFrame for editing
    quant_df = pd.DataFrame({
        'Substance': substances,
        'Mass (g)': [f"{mass:.4f}" if mass > 0 else "" for mass in current_masses],
        'Volume (mL)': [f"{vol:.2f}" if vol > 0 else "" for vol in current_volumes],
        'Units': ['g', 'mL', 'g', 'mL', 'g', 'g', 'mL', 'g'],
        'Status': ['✅' if mass > 0 or vol > 0 else '⚪' for mass, vol in zip(current_masses, current_volumes)]
    })
    
    st.write("**Current Measurements:**")
    edited_quant_df = st.data_editor(
        quant_df, 
        hide_index=True, 
        key="quantitative_table",
        column_config={
            "Mass (g)": st.column_config.NumberColumn(
                "Mass (g)",
                help="Enter mass in grams",
                min_value=0.0,
                step=0.0001,
                format="%.4f"
            ),
            "Volume (mL)": st.column_config.NumberColumn(
                "Volume (mL)", 
                help="Enter volume in milliliters",
                min_value=0.0,
                step=0.01,
                format="%.2f"
            ),
            "Units": st.column_config.TextColumn(
                "Units",
                disabled=True
            ),
            "Status": st.column_config.TextColumn(
                "Status",
                disabled=True
            )
        }
    )
    
    # Update experiment data button
    if st.button("💾 Update Experiment Data", key="update_quant_data"):
        update_experiment_from_table(edited_quant_df, experiment_id)
        st.success("Experiment data updated!")
        st.rerun()
    
    # Quick entry forms
    st.subheader("🚀 Quick Data Entry")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Mass Measurements**")
        with st.form("mass_entry"):
            compound = st.selectbox("Compound", [
                "HAuCl₄·3H₂O", "TOAB", "PhCH₂CH₂SH", "NaBH₄", "Au₂₅ nanoparticles"
            ])
            mass_value = st.number_input("Mass (g)", min_value=0.0, step=0.0001, format="%.4f")
            
            if st.form_submit_button("Record Mass"):
                record_measurement(experiment_id, compound, mass_value, "g", "mass")
                st.success(f"Recorded {compound}: {mass_value}g")
                st.rerun()
    
    with col2:
        st.write("**Volume Measurements**")
        with st.form("volume_entry"):
            liquid = st.selectbox("Liquid", [
                "nanopure water", "toluene", "ice-cold nanopure water"
            ])
            volume_value = st.number_input("Volume (mL)", min_value=0.0, step=0.01, format="%.2f")
            
            if st.form_submit_button("Record Volume"):
                record_measurement(experiment_id, liquid, volume_value, "mL", "volume")
                st.success(f"Recorded {liquid}: {volume_value}mL")
                st.rerun()
def render_qualitative_panel(experiment_id: str):
    """Render qualitative observations panel"""
    st.subheader("🔍 Qualitative Observations")
    
    # Initialize qualitative data in session state
    if 'qualitative_data' not in st.session_state:
        st.session_state.qualitative_data = {
            'Step 4 - Gold solution': {'color': '', 'clarity': '', 'notes': ''},
            'Step 8 - Two-phase system': {'separation': '', 'color_organic': '', 'color_aqueous': '', 'notes': ''},
            'Step 20 - Color changes': {'initial_color': '', 'intermediate_color': '', 'final_color': '', 'notes': ''},
            'Step 25 - NaBH₄ addition': {'immediate_change': '', 'final_appearance': '', 'notes': ''},
            'Step 27 - Next day': {'organic_phase': '', 'aqueous_phase': '', 'precipitation': '', 'notes': ''}
        }
    
    # Create observation forms
    for step_name, observations in st.session_state.qualitative_data.items():
        with st.expander(f"📝 {step_name}", expanded=False):
            step_key = step_name.replace(' ', '_').replace('-', '_').lower()
            
            if 'Gold solution' in step_name:
                col1, col2 = st.columns(2)
                with col1:
                    observations['color'] = st.text_input("Color", value=observations.get('color', ''), key=f"{step_key}_color")
                with col2:
                    observations['clarity'] = st.selectbox("Clarity", ["Clear", "Cloudy", "Precipitated"], 
                                                         index=0 if not observations.get('clarity') else ["Clear", "Cloudy", "Precipitated"].index(observations['clarity']),
                                                         key=f"{step_key}_clarity")
            
            elif 'Two-phase' in step_name:
                col1, col2, col3 = st.columns(3)
                with col1:
                    observations['separation'] = st.selectbox("Separation Quality", ["Poor", "Good", "Excellent"],
                                                            index=0 if not observations.get('separation') else ["Poor", "Good", "Excellent"].index(observations['separation']),
                                                            key=f"{step_key}_separation")
                with col2:
                    observations['color_organic'] = st.text_input("Organic Phase Color", value=observations.get('color_organic', ''), key=f"{step_key}_org_color")
                with col3:
                    observations['color_aqueous'] = st.text_input("Aqueous Phase Color", value=observations.get('color_aqueous', ''), key=f"{step_key}_aq_color")
            
            elif 'Color changes' in step_name:
                col1, col2, col3 = st.columns(3)
                with col1:
                    observations['initial_color'] = st.text_input("Initial Color", value=observations.get('initial_color', ''), key=f"{step_key}_init_color")
                with col2:
                    observations['intermediate_color'] = st.text_input("Intermediate Color", value=observations.get('intermediate_color', ''), key=f"{step_key}_inter_color")
                with col3:
                    observations['final_color'] = st.text_input("Final Color", value=observations.get('final_color', ''), key=f"{step_key}_final_color")
            
            elif 'NaBH₄' in step_name:
                col1, col2 = st.columns(2)
                with col1:
                    observations['immediate_change'] = st.text_area("Immediate Changes", value=observations.get('immediate_change', ''), height=68, key=f"{step_key}_immediate")
                with col2:
                    observations['final_appearance'] = st.text_area("Final Appearance", value=observations.get('final_appearance', ''), height=68, key=f"{step_key}_final_app")
            
            elif 'Next day' in step_name:
                col1, col2, col3 = st.columns(3)
                with col1:
                    observations['organic_phase'] = st.text_input("Organic Phase", value=observations.get('organic_phase', ''), key=f"{step_key}_org_phase")
                with col2:
                    observations['aqueous_phase'] = st.text_input("Aqueous Phase", value=observations.get('aqueous_phase', ''), key=f"{step_key}_aq_phase")
                with col3:
                    observations['precipitation'] = st.text_input("Precipitation", value=observations.get('precipitation', ''), key=f"{step_key}_precip")
            
            # Notes for all steps
            observations['notes'] = st.text_area("Additional Notes", value=observations.get('notes', ''), height=80, key=f"{step_key}_notes")
    
    # Save observations
    if st.button("💾 Save Qualitative Observations", key="save_qual_obs"):
        save_qualitative_observations(experiment_id, st.session_state.qualitative_data)
        st.success("Qualitative observations saved!")
def render_visualization_panel(exp_data: dict):
    """Render data visualization panel"""
    st.subheader("📈 Data Visualization & Analysis")
    
    # Create mass vs theoretical comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Mass Distribution**")
        masses = {
            'HAuCl₄·3H₂O': exp_data.get('mass_gold', 0),
            'TOAB': exp_data.get('mass_toab', 0),
            'PhCH₂CH₂SH': exp_data.get('mass_sulfur', 0),
            'NaBH₄': exp_data.get('mass_nabh4', 0),
            'Final Au₂₅': exp_data.get('mass_final', 0)
        }
        
        # Filter out zero values
        masses_filtered = {k: v for k, v in masses.items() if v > 0}
        
        if masses_filtered:
            fig_mass = px.bar(
                x=list(masses_filtered.keys()),
                y=list(masses_filtered.values()),
                title="Compound Masses (g)",
                labels={'x': 'Compounds', 'y': 'Mass (g)'}
            )
            fig_mass.update_layout(height=400)
            st.plotly_chart(fig_mass, use_container_width=True)
        else:
            st.info("No mass data available for visualization")
    
    with col2:
        st.write("**Volume Distribution**")
        volumes = {
            'Nanopure (RT)': exp_data.get('volume_nanopure_rt', 0),
            'Toluene': exp_data.get('volume_toluene', 0),
            'Nanopure (Cold)': exp_data.get('volume_nanopure_cold', 0)
        }
        
        # Filter out zero values
        volumes_filtered = {k: v for k, v in volumes.items() if v > 0}
        
        if volumes_filtered:
            fig_vol = px.pie(
                values=list(volumes_filtered.values()),
                names=list(volumes_filtered.keys()),
                title="Volume Distribution (mL)"
            )
            fig_vol.update_layout(height=400)
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info("No volume data available for visualization")
    
    # Stoichiometry analysis
    st.subheader("🧮 Stoichiometry Analysis")
    
    if exp_data.get('mass_gold', 0) > 0:
        # Calculate theoretical amounts
        gold_mass = exp_data['mass_gold']
        
        # Molecular weights
        mw_haucl4_3h2o = 393.83
        mw_sulfur = 138.21
        mw_nabh4 = 37.83
        
        # Calculate moles and theoretical masses
        moles_gold = gold_mass / mw_haucl4_3h2o
        theoretical_sulfur = moles_gold * 3 * mw_sulfur  # 3 equivalents
        theoretical_nabh4 = moles_gold * 10 * mw_nabh4   # 10 equivalents
        
        # Create comparison table
        comparison_data = {
            'Compound': ['PhCH₂CH₂SH', 'NaBH₄'],
            'Theoretical (g)': [theoretical_sulfur, theoretical_nabh4],
            'Actual (g)': [exp_data.get('mass_sulfur', 0), exp_data.get('mass_nabh4', 0)],
            'Ratio': [
                exp_data.get('mass_sulfur', 0) / theoretical_sulfur if theoretical_sulfur > 0 else 0,
                exp_data.get('mass_nabh4', 0) / theoretical_nabh4 if theoretical_nabh4 > 0 else 0
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df['Ratio'] = comparison_df['Ratio'].round(3)
        
        st.dataframe(comparison_df, hide_index=True, use_container_width=True)
        
        # Stoichiometry chart
        fig_stoich = go.Figure()
        fig_stoich.add_trace(go.Bar(
            name='Theoretical',
            x=comparison_data['Compound'],
            y=comparison_data['Theoretical (g)'],
            marker_color='lightblue'
        ))
        fig_stoich.add_trace(go.Bar(
            name='Actual',
            x=comparison_data['Compound'], 
            y=comparison_data['Actual (g)'],
            marker_color='darkblue'
        ))
        
        fig_stoich.update_layout(
            title='Theoretical vs Actual Amounts',
            xaxis_title='Compounds',
            yaxis_title='Mass (g)',
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig_stoich, use_container_width=True)
def render_export_panel(exp_data: dict, experiment_id: str):
    """Render data export panel"""
    st.subheader("📋 Data Export & Reporting")
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Export Formats**")
        
        if st.button("📊 Export to CSV", key="export_csv"):
            csv_data = create_csv_export(exp_data, experiment_id)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name=f"experiment_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        if st.button("📄 Generate Report", key="generate_report"):
            report_data = create_experiment_report(exp_data, experiment_id)
            st.download_button(
                label="Download Report",
                data=report_data,
                file_name=f"report_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with col2:
        st.write("**Data Summary**")
        
        # Data completeness metrics
        total_fields = 8  # Number of data fields
        completed_fields = sum([
            1 if exp_data.get('mass_gold', 0) > 0 else 0,
            1 if exp_data.get('mass_toab', 0) > 0 else 0,
            1 if exp_data.get('mass_sulfur', 0) > 0 else 0,
            1 if exp_data.get('mass_nabh4', 0) > 0 else 0,
            1 if exp_data.get('mass_final', 0) > 0 else 0,
            1 if exp_data.get('volume_toluene', 0) > 0 else 0,
            1 if exp_data.get('volume_nanopure_rt', 0) > 0 else 0,
            1 if exp_data.get('volume_nanopure_cold', 0) > 0 else 0,
        ])
        
        completeness = (completed_fields / total_fields) * 100
        
        st.metric("Data Completeness", f"{completeness:.1f}%")
        st.progress(completeness / 100)
        
        st.metric("Total Measurements", completed_fields)
        st.metric("Experiment Status", exp_data.get('status', 'unknown').title())
# Helper functions
def update_experiment_from_table(df: pd.DataFrame, experiment_id: str):
    """Update experiment data from edited table"""
    # Map table rows to API fields
    substance_mapping = {
        'HAuCl₄·3H₂O': ('mass_gold', 'mass'),
        'TOAB': ('mass_toab', 'mass'),
        'PhCH₂CH₂SH': ('mass_sulfur', 'mass'),
        'NaBH₄': ('mass_nabh4', 'mass'),
        'Final mass of Au₂₅ nanoparticles': ('mass_final', 'mass'),
        'Water (for gold)': ('volume_nanopure_rt', 'volume'),
        'Toluene': ('volume_toluene', 'volume'),
        'Ice-cold Nanopure water (for NaBH₄)': ('volume_nanopure_cold', 'volume')
    }
    
    for _, row in df.iterrows():
        substance = row['Substance']
        if substance in substance_mapping:
            field_name, data_type = substance_mapping[substance]
            
            if data_type == 'mass' and row['Mass (g)']:
                try:
                    value = float(row['Mass (g)'])
                    record_measurement(experiment_id, substance, value, 'g', 'mass')
                except:
                    pass
            elif data_type == 'volume' and row['Volume (mL)']:
                try:
                    value = float(row['Volume (mL)'])
                    record_measurement(experiment_id, substance, value, 'mL', 'volume')
                except:
                    pass
def record_measurement(experiment_id: str, compound: str, value: float, units: str, data_type: str):
    """Record a measurement via API"""
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
def save_qualitative_observations(experiment_id: str, observations: dict):
    """Save qualitative observations via API"""
    # Convert observations to a format suitable for API
    obs_text = ""
    for step, data in observations.items():
        obs_text += f"{step}:\n"
        for key, value in data.items():
            if value:
                obs_text += f"  {key}: {value}\n"
        obs_text += "\n"
    
    # Update experiment with observations
    try:
        response = requests.put(
            f"{API_BASE_URL}/experiments/{experiment_id}/observations",
            json={"observations": obs_text}
        )
        return response.status_code == 200
    except:
        return False
def create_csv_export(exp_data: dict, experiment_id: str) -> str:
    """Create CSV export of experiment data"""
    import io
    
    output = io.StringIO()
    
    # Write header
    output.write(f"Experiment ID,{experiment_id}\n")
    output.write(f"Created,{exp_data.get('created_at', '')}\n")
    output.write(f"Status,{exp_data.get('status', '')}\n")
    output.write("\n")
    
    # Write quantitative data
    output.write("Substance,Mass (g),Volume (mL)\n")
    substances = [
        ('HAuCl₄·3H₂O', exp_data.get('mass_gold', 0), ''),
        ('TOAB', exp_data.get('mass_toab', 0), ''),
        ('PhCH₂CH₂SH', exp_data.get('mass_sulfur', 0), ''),
        ('NaBH₄', exp_data.get('mass_nabh4', 0), ''),
        ('Final Au₂₅', exp_data.get('mass_final', 0), ''),
        ('Nanopure (RT)', '', exp_data.get('volume_nanopure_rt', 0)),
        ('Toluene', '', exp_data.get('volume_toluene', 0)),
        ('Nanopure (Cold)', '', exp_data.get('volume_nanopure_cold', 0))
    ]
    
    for substance, mass, volume in substances:
        output.write(f"{substance},{mass},{volume}\n")
    
    return output.getvalue()
def create_experiment_report(exp_data: dict, experiment_id: str) -> str:
    """Create detailed experiment report"""
    report = f"""
EXPERIMENT REPORT
================
Experiment ID: {experiment_id}
Created: {exp_data.get('created_at', '')}
Last Updated: {exp_data.get('updated_at', '')}
Status: {exp_data.get('status', '').title()}
Current Step: {exp_data.get('step_num', 0)}/12
QUANTITATIVE DATA
----------------
HAuCl₄·3H₂O: {exp_data.get('mass_gold', 0):.4f} g
TOAB: {exp_data.get('mass_toab', 0):.4f} g  
PhCH₂CH₂SH: {exp_data.get('mass_sulfur', 0):.4f} g
NaBH₄: {exp_data.get('mass_nabh4', 0):.4f} g
Final Au₂₅: {exp_data.get('mass_final', 0):.4f} g
Nanopure water (RT): {exp_data.get('volume_nanopure_rt', 0):.2f} mL
Toluene: {exp_data.get('volume_toluene', 0):.2f} mL
Nanopure water (Cold): {exp_data.get('volume_nanopure_cold', 0):.2f} mL
SAFETY STATUS
------------
Current Status: {exp_data.get('safety_status', 'unknown').title()}
OBSERVATIONS
-----------
{exp_data.get('observations', 'No observations recorded')}
Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    return report
