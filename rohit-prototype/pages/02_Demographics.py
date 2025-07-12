import streamlit as st
import plotly.express as px

st.title('Demographic Analysis')

if not st.session_state.data.empty:
    col1, col2 = st.columns(2)

    with col1:
        # Age Distribution
        st.header('Age Distribution')
        fig_age = px.histogram(st.session_state.data, x='Age', nbins=20, title='Age Distribution')
        st.plotly_chart(fig_age)

        # Average Sales by Age Group
        st.session_state.data['Age_Group'] = pd.cut(st.session_state.data['Age'], 
                                                  bins=[0, 25, 35, 45, 55, 120], 
                                                  labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        avg_sales_age = st.session_state.data.groupby('Age_Group')['Sales'].mean().reset_index()
        fig_avg_sales = px.bar(avg_sales_age, x='Age_Group', y='Sales', 
                              title='Average Sales by Age Group')
        st.plotly_chart(fig_avg_sales)

    with col2:
        # Region Distribution
        st.header('Regional Demographics')
        fig_region = px.scatter(st.session_state.data, x='Age', y='Sales', 
                               color='Region', size='Sales',
                               title='Age vs Sales by Region')
        st.plotly_chart(fig_region)

        # Category Distribution
        category_dist = st.session_state.data['Category'].value_counts()
        fig_category = px.pie(values=category_dist.values, 
                             names=category_dist.index,
                             title='Distribution by Category')
        st.plotly_chart(fig_category)
else:
    st.info('Please add some data in the main page first!')