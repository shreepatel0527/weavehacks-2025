import streamlit as st
import plotly.express as px

st.title('Sales Analysis')

if not st.session_state.data.empty:
    # Sales by Region
    st.header('Sales by Region')
    fig_region = px.pie(st.session_state.data, values='Sales', names='Region', title='Sales Distribution by Region')
    st.plotly_chart(fig_region)

    # Sales Trend
    st.header('Sales Trend')
    sales_trend = st.session_state.data.groupby('Date')['Sales'].sum().reset_index()
    fig_trend = px.line(sales_trend, x='Date', y='Sales', title='Daily Sales Trend')
    st.plotly_chart(fig_trend)

    # Sales by Category
    st.header('Sales by Category')
    fig_category = px.bar(st.session_state.data, x='Category', y='Sales', title='Sales by Category')
    st.plotly_chart(fig_category)
else:
    st.info('Please add some data in the main page first!')