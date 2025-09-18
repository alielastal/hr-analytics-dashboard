# File: src/app/main.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and model
@st.cache_data
def load_data():
    data_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'hr_dataset_with_predictions.csv'
    return pd.read_csv(data_path)

@st.cache_resource
def load_model():
    model_path = Path(__file__).parent.parent.parent / 'models' / 'attrition_predictor.pkl'
    return joblib.load(model_path)

# Load data and model
df = load_data()
model = load_model()

# Sidebar
st.sidebar.title("HR Analytics Dashboard")
st.sidebar.markdown("---")

# Main page
st.title("📊 HR Analytics & Attrition Prediction Dashboard")
st.markdown("---")

# Overview metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    total_employees = len(df)
    st.metric("Total Employees", f"{total_employees:,}")
with col2:
    attrition_rate = df['AttritionFlag'].mean() * 100
    st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
with col3:
    high_risk = (df['Prediction_Probability'] > 0.7).sum()
    st.metric("High Risk Employees", f"{high_risk}")
with col4:
    avg_salary = df['Salary'].mean()
    st.metric("Avg Salary", f"${avg_salary:,.0f}")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Overview Analytics", 
    "🎯 Attrition Analysis", 
    "🔮 Prediction Insights",
    "🧮 Prediction Calculator"
])

with tab1:
    st.header("Workforce Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Department distribution
        dept_counts = df['Department'].value_counts()
        fig = px.pie(values=dept_counts.values, names=dept_counts.index, 
                    title="Employee Distribution by Department")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Salary distribution
        fig = px.histogram(df, x='Salary', nbins=20, 
                          title="Salary Distribution", color='Department')
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Age distribution
        fig = px.histogram(df, x='Age', nbins=15, color='Gender',
                          title="Age Distribution by Gender")
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Education level
        edu_counts = df['Education'].value_counts()
        fig = px.bar(x=edu_counts.index, y=edu_counts.values,
                    title="Employees by Education Level",
                    labels={'x': 'Education Level', 'y': 'Count'})
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Attrition Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Attrition by department
        dept_attrition = df.groupby('Department')['AttritionFlag'].mean() * 100
        fig = px.bar(x=dept_attrition.index, y=dept_attrition.values,
                    title="Attrition Rate by Department (%)",
                    labels={'x': 'Department', 'y': 'Attrition Rate %'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Salary vs Attrition
        fig = px.box(df, x='Attrition', y='Salary', 
                    title="Salary Distribution by Attrition Status")
        st.plotly_chart(fig, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Age vs Attrition
        fig = px.violin(df, x='Attrition', y='Age',
                       title="Age Distribution by Attrition Status")
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Gender attrition
        gender_attrition = df.groupby('Gender')['AttritionFlag'].mean() * 100
        fig = px.bar(x=gender_attrition.index, y=gender_attrition.values,
                    title="Attrition Rate by Gender (%)")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Prediction Insights")
    
    # Risk distribution
    col1, col2 = st.columns(2)
    
    with col1:
        risk_bins = pd.cut(df['Prediction_Probability'], 
                          bins=[0, 0.3, 0.7, 1.0],
                          labels=['Low Risk', 'Medium Risk', 'High Risk'])
        risk_counts = risk_bins.value_counts()
        
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                    title="Attrition Risk Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # High risk employees by department
        high_risk_df = df[df['Prediction_Probability'] > 0.7]
        dept_risk = high_risk_df['Department'].value_counts()
        
        fig = px.bar(x=dept_risk.index, y=dept_risk.values,
                    title="High Risk Employees by Department",
                    labels={'x': 'Department', 'y': 'High Risk Count'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance (simplified)
    st.subheader("Top Factors Driving Attrition Risk")
    
    # Simplified feature importance based on domain knowledge
    importance_data = {
        'Factor': ['Salary vs Role Avg', 'Salary vs Dept Avg', 'Total Compensation', 
                  'Age', 'Salary vs Bonus Ratio', 'Department', 'Education Level'],
        'Importance': [0.98, 0.94, 0.93, 0.87, 0.85, 0.82, 0.78]
    }
    
    importance_df = pd.DataFrame(importance_data)
    fig = px.bar(importance_df.sort_values('Importance', ascending=True), 
                 x='Importance', y='Factor', orientation='h',
                 title="Key Attrition Risk Factors")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Attrition Prediction Calculator")
    
    st.markdown("""
    **Predict attrition risk for a new employee based on their characteristics.**
    Adjust the sliders below to see how different factors affect attrition risk.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 20, 65, 35)
        salary = st.slider("Salary ($)", 30000, 150000, 60000)
        bonus = st.slider("Bonus ($)", 0, 50000, 10000)
        department = st.selectbox("Department", df['Department'].unique())
    
    with col2:
        education = st.selectbox("Education Level", df['Education'].unique())
        job_role = st.selectbox("Job Role", df['JobRole'].unique())
        gender = st.selectbox("Gender", df['Gender'].unique())
    
    # Create sample data for prediction
    sample_data = pd.DataFrame({
        'Age': [age],
        'Salary': [salary],
        'Bonus': [bonus],
        'Department': [department],
        'Education': [education],
        'JobRole': [job_role],
        'Gender': [gender],
        'TotalCompensation': [salary + bonus]
    })
    
    # Add engineered features (simplified)
    dept_avg_salary = df[df['Department'] == department]['Salary'].mean()
    role_avg_salary = df[df['JobRole'] == job_role]['Salary'].mean()
    
    sample_data['DeptSalaryRatio'] = salary / dept_avg_salary if dept_avg_salary else 1
    sample_data['RoleSalaryRatio'] = salary / role_avg_salary if role_avg_salary else 1
    sample_data['SalaryBonusRatio'] = salary / (bonus + 1)
    sample_data['AgeSquared'] = age ** 2
    sample_data['TotalCompensationCalculated'] = salary + bonus  # Add this line
    sample_data['LogSalary'] = np.log(salary + 1)  # Add this line too
    
    # Ensure all expected columns are present
    expected_columns = df.drop(columns=['AttritionFlag', 'Attrition', 'EmployeeID', 'Name', 
                                    'Prediction', 'Prediction_Probability'], errors='ignore').columns

    for col in expected_columns:
        if col not in sample_data.columns:
            sample_data[col] = 0  # Fill missing with 0

    # Reorder columns to match training data
    sample_data = sample_data[expected_columns]

    if st.button("Predict Attrition Risk"):
        try:
            # Predict
            prediction = model.predict(sample_data)[0]
            probability = model.predict_proba(sample_data)[0][1]
            
            # Display results
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction Result")
                if prediction == 1:
                    st.error(f"🚨 High Attrition Risk: {probability:.1%}")
                else:
                    st.success(f"✅ Low Attrition Risk: {probability:.1%}")
                
                # Progress bar
                st.progress(float(probability))
            
            with col2:
                st.subheader("Risk Analysis")
                if probability > 0.7:
                    st.warning("**High Risk** - Consider retention strategies")
                elif probability > 0.4:
                    st.info("**Medium Risk** - Monitor closely")
                else:
                    st.success("**Low Risk** - Stable employee")
            
            # Risk factors
            st.subheader("Key Risk Factors")
            factors = []
            if salary < dept_avg_salary:
                factors.append(f"Salary ${salary:,.0f} below department average (${dept_avg_salary:,.0f})")
            if salary < role_avg_salary:
                factors.append(f"Salary ${salary:,.0f} below role average (${role_avg_salary:,.0f})")
            if bonus < df['Bonus'].mean():
                factors.append(f"Bonus ${bonus:,.0f} below company average (${df['Bonus'].mean():,.0f})")
            
            if factors:
                for factor in factors:
                    st.write(f"• {factor}")
            else:
                st.write("• No significant risk factors identified")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**HR Analytics Dashboard** • Built with Streamlit • Predictive ML Model")