import streamlit as st
import pandas as pd

st.title('Data Management')

if not st.session_state.data.empty:
    # Data Filtering
    st.header('Filter Data')
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_region = st.multiselect('Select Regions', 
                                        options=st.session_state.data['Region'].unique())
    with col2:
        selected_category = st.multiselect('Select Categories', 
                                          options=st.session_state.data['Category'].unique())
    with col3:
        min_sales = st.number_input('Minimum Sales', 
                                   value=float(st.session_state.data['Sales'].min()))

    # Apply filters
    filtered_data = st.session_state.data.copy()
    if selected_region:
        filtered_data = filtered_data[filtered_data['Region'].isin(selected_region)]
    if selected_category:
        filtered_data = filtered_data[filtered_data['Category'].isin(selected_category)]
    filtered_data = filtered_data[filtered_data['Sales'] >= min_sales]

    # Display filtered data
    st.header('Filtered Data')
    st.dataframe(filtered_data)

    # Data Export
    st.header('Export Data')
    if st.button('Download CSV'):
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download filtered data as CSV",
            data=csv,
            file_name='filtered_data.csv',
            mime='text/csv',
        )

    # Data Statistics
    st.header('Data Statistics')
    st.write(filtered_data.describe())
else:
    st.info('Please add some data in the main page first!')