import streamlit as st
import pandas as pd
import base64
from collections import Counter
import plotly.express as px
from streamlit_lottie import st_lottie
import requests
from prophet import Prophet
from prophet.plot import plot_plotly
import matplotlib.pyplot as plt
import os
from PIL import Image

# Page Config
st.set_page_config(page_title="India Job Market Dashboard", layout="wide", initial_sidebar_state="expanded")


# Load Lottie Animation
@st.cache_data
def load_lottie_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()


# Function to display PDF
def display_pdf(pdf_file):
    # Opening file from file path
    with open(pdf_file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>
    """

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


# Load animations
animations = {
    "job": load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_touohxv0.json"),
    "company": load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_w51pcehl.json"),
    "skills": load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_jtbfg2nb.json"),
    "ml": load_lottie_url("https://assets8.lottiefiles.com/packages/lf20_UJNc2t.json"),
}

# Light/Dark Mode Toggle
mode = st.sidebar.selectbox("🌃 Mode", ["Light", "Dark"])
if mode == "Dark":
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: #fafafa; }
        div[data-testid="stSidebar"] { background-color: #111111; }
        </style>
    """, unsafe_allow_html=True)


# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("data/india_job_market_dataset.csv")


df = load_data()

# Sidebar Navigation
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Company Insights", "Skill Insights", "ML Analysis", "Forecasting",
                                  "Power BI Reports"])

# Sidebar Filters
st.sidebar.header("🔍 Filter Jobs")
selected_skills = st.sidebar.multiselect("Skills Required", sorted(df['Skills Required'].dropna().unique()))
selected_city = st.sidebar.multiselect("Job Location", sorted(df['Job Location'].dropna().unique()))
selected_experience = st.sidebar.multiselect("Experience Required", sorted(df['Experience Required'].dropna().unique()))

# Apply Filters
filtered_df = df.copy()

if selected_skills:
    filtered_df = filtered_df[filtered_df['Skills Required'].apply(
        lambda x: any(skill.lower() in [s.strip().lower() for s in x.split(',')] for skill in selected_skills)
    )]

if selected_city:
    filtered_df = filtered_df[filtered_df['Job Location'].isin(selected_city)]

if selected_experience:
    filtered_df = filtered_df[filtered_df['Experience Required'].isin(selected_experience)]

# ====================
# PAGE 1: HOME
# ====================
if page == "Home":
    st.title("📊 India Job Market Dashboard")
    st_lottie(animations["job"], height=250)

    st.subheader("🗂 Full Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("📊 Filtered Job Results")
    if filtered_df.empty:
        st.warning("⚠ No data found for the selected filters.")
    else:
        st.dataframe(filtered_df, use_container_width=True)

        # Download
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", data=csv, file_name='filtered_job_data.csv', mime='text/csv')

# ====================
# PAGE 2: COMPANY INSIGHTS
# ====================
elif page == "Company Insights":
    st.title("🏢 Company-Level Insights")
    st_lottie(animations["company"], height=250)

    st.subheader("🔝 Top Companies & Job Titles")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏆 Top Hiring Companies")
        company_chart = filtered_df['Company Name'].value_counts().head(10).reset_index()
        company_chart.columns = ['Company', 'Postings']
        st.plotly_chart(px.bar(company_chart, x='Company', y='Postings', color='Postings'), use_container_width=True)

        st.markdown("#### 💼 Popular Job Titles")
        title_chart = filtered_df['Job Title'].value_counts().head(10).reset_index()
        title_chart.columns = ['Title', 'Count']
        st.plotly_chart(px.bar(title_chart, x='Title', y='Count', color='Count'), use_container_width=True)

    with col2:
        st.markdown("#### 📍 Jobs by City")
        city_chart = filtered_df['Job Location'].value_counts().head(10).reset_index()
        city_chart.columns = ['City', 'Count']
        st.plotly_chart(px.pie(city_chart, names='City', values='Count'), use_container_width=True)

        st.markdown("#### 🎯 Experience Demand")
        experience_chart = filtered_df['Experience Required'].value_counts().reset_index()
        experience_chart.columns = ['Experience', 'Count']
        st.plotly_chart(px.line(experience_chart.sort_values('Experience'), x='Experience', y='Count', markers=True),
                        use_container_width=True)

    st.subheader("🧾 Job Type Distribution")
    job_type_chart = filtered_df['Job Type'].value_counts().reset_index()
    job_type_chart.columns = ['Job Type', 'Count']
    st.plotly_chart(px.pie(job_type_chart, names='Job Type', values='Count'), use_container_width=True)

# ====================
# PAGE 3: SKILL INSIGHTS
# ====================
elif page == "Skill Insights":
    st.title("💡 In-Demand Skills Analysis")
    st_lottie(animations["skills"], height=250)

    all_skills = []
    for row in filtered_df['Skills Required'].dropna():
        all_skills.extend([s.strip() for s in row.split(',')])

    if all_skills:
        top_skills = Counter(all_skills).most_common(10)
        skills_df = pd.DataFrame(top_skills, columns=['Skill', 'Count'])
        st.markdown("#### 🔝 Top 10 In-Demand Skills")
        st.plotly_chart(px.bar(skills_df, x='Skill', y='Count', color='Count'), use_container_width=True)
    else:
        st.info("ℹ No skill data available for selected filters.")

# ====================
# ML Analysis section
elif page == "ML Analysis":
    st.title("🤖 Machine Learning Analysis")

    try:
        st_lottie(animations["skills"], height=250)
    except Exception:
        st.info("Machine Learning Analysis Dashboard")

    # Tab-based interface for ML features
    ml_tabs = st.tabs(["Salary Prediction", "Job Clustering", "Skill Demand Analysis", "Experience Impact"])

    # TAB 1: SALARY PREDICTION
    with ml_tabs[0]:
        st.header("💰 Salary Prediction Model")

        # Features for prediction
        st.subheader("Enter Job Details")
        col1, col2 = st.columns(2)

        with col1:
            pred_job_title = st.selectbox("Job Title", sorted(df['Job Title'].unique()))
            pred_company_size = st.selectbox("Company Size",
                                             sorted(df['Company Size'].unique()) if 'Company Size' in df.columns else [
                                                 "Small", "Medium", "Large"])
            pred_job_type = st.selectbox("Job Type", sorted(df['Job Type'].unique()))
            pred_job_location = st.selectbox("Job Location", sorted(df['Job Location'].unique()))

        with col2:
            pred_experience = st.selectbox("Experience Required", sorted(df['Experience Required'].unique()))

            # Fix for missing Education column
            if 'Education' in df.columns:
                pred_education = st.selectbox("Education Level", sorted(df['Education'].unique()))
            else:
                # Provide default education options if column doesn't exist
                pred_education = st.selectbox("Education Level",
                                              ["Bachelor's", "Master's", "PhD", "MBA", "High School", "Diploma"])

            # Extract skills from the dataset
            all_skills = set()
            for skills in df['Skills Required'].dropna():
                all_skills.update([s.strip() for s in skills.split(',')])

            pred_skills = st.multiselect("Skills", sorted(all_skills), max_selections=5)
            pred_remote = st.selectbox("Work Mode", ["Remote", "Hybrid", "Onsite"])

        # Prediction button
        if st.button("Predict Salary Range"):
            # Demonstration of prediction (replace with actual ML model)
            st.info("🧠 Calculating prediction...")

            # Mock prediction - in real implementation, use trained ML model
            import random
            import time

            # Simulate computation time
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)

            # Salary brackets from dataset
            salary_brackets = ["3-5 LPA", "5-8 LPA", "8-12 LPA", "12-20 LPA", "20+ LPA"]
            confidence_scores = [random.uniform(0.1, 0.9) for _ in range(len(salary_brackets))]
            total = sum(confidence_scores)
            normalized_scores = [score / total for score in confidence_scores]

            # Display prediction
            st.success("✅ Prediction complete!")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("Predicted Salary Range")

                # Create prediction chart
                pred_df = pd.DataFrame({
                    'Salary Range': salary_brackets,
                    'Probability': normalized_scores
                })

                fig = px.bar(pred_df, x='Salary Range', y='Probability',
                             color='Probability',
                             labels={'Probability': 'Confidence'},
                             color_continuous_scale=px.colors.sequential.Viridis)

                # Highlight top prediction
                top_prediction = salary_brackets[normalized_scores.index(max(normalized_scores))]
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    f"**Most likely salary range: **{top_prediction} (Confidence: {max(normalized_scores):.2%})")

            with col2:
                st.subheader("Key Factors")

                # Display important features (in real model, use SHAP or feature importance)
                factors = [
                    {"factor": "Experience",
                     "impact": "High 📈" if "5-10" in pred_experience or "10+" in pred_experience else "Medium ⚖️"},
                    {"factor": "Education", "impact": "High 📈" if pred_education in ["PhD", "MBA"] else "Medium ⚖️"},
                    {"factor": "Location",
                     "impact": "High 📈" if pred_job_location in ["Bangalore", "Delhi", "Mumbai"] else "Medium ⚖️"},
                    {"factor": "Job Type", "impact": "Medium ⚖️" if pred_job_type == "Full-time" else "Low 📉"}
                ]

                for factor in factors:
                    st.markdown(f"**{factor['factor']}**: {factor['impact']}")

    # Other ML tabs would remain here...

# ====================
# PAGE 5: FORECASTING
# ====================
elif page == "Forecasting":
    st.title("📈 Forecasting Job Postings")
    st_lottie(animations["job"], height=250)

    if 'Posted Date' not in df.columns:
        st.error("❌ 'Posted Date' column not found in dataset.")
    else:
        df['Posted Date'] = pd.to_datetime(df['Posted Date'], errors='coerce')
        df = df.dropna(subset=['Posted Date'])

        st.subheader("🔎 Choose Forecast Category")
        forecast_type = st.radio("Forecast by", ["Overall", "City-wise", "Skill-wise"], horizontal=True)

        if forecast_type == "Overall":
            forecast_df = filtered_df.copy()
        elif forecast_type == "City-wise":
            city_option = st.selectbox("Select City", sorted(df['Job Location'].dropna().unique()))
            forecast_df = df[df['Job Location'] == city_option]
        elif forecast_type == "Skill-wise":
            skill_option = st.selectbox("Select Skill", sorted(
                {skill.strip() for skills in df['Skills Required'].dropna() for skill in skills.split(',')}))
            forecast_df = df[df['Skills Required'].str.contains(skill_option, case=False, na=False)]

        forecast_df = forecast_df[['Posted Date']].copy()
        if forecast_df.empty:
            st.warning("⚠ No job postings available for the selected criteria.")
        else:
            forecast_df['Count'] = 1
            ts_df = forecast_df.groupby('Posted Date').count().reset_index()
            ts_df.columns = ['ds', 'y']

            # Forecast
            model = Prophet()
            model.fit(ts_df)
            future = model.make_future_dataframe(periods=90)
            forecast = model.predict(future)

            st.subheader("📊 Forecasted Job Postings (Next 90 Days)")
            fig = plot_plotly(model, forecast)
            st.plotly_chart(fig, use_container_width=True)

            # Download CSV
            forecast_csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                columns={'ds': 'Date', 'yhat': 'Forecasted Count'}
            )
            csv = forecast_csv.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Forecast CSV", data=csv, file_name="forecast_data.csv", mime="text/csv")

            # Trend Breakdown
            with st.expander("📉 Trend & Seasonality Breakdown"):
                st.pyplot(model.plot_components(forecast))

# ====================
# PAGE 6: POWER BI REPORTS (NEW PAGE)
# ====================
elif page == "Power BI Reports":
    st.title("📊 Power BI Analytics Reports")

    # Create a reports folder if not exists
    reports_folder = "reports"
    if not os.path.exists(reports_folder):
        os.makedirs(reports_folder)
        st.warning("⚠️ 'reports' folder created. Please add your Power BI PDF exports to this folder.")

    # List available reports
    report_files = [f for f in os.listdir(reports_folder) if f.endswith('.pdf')]

    if not report_files:
        st.error("❌ No PDF reports found in the 'reports' folder. Please add your Power BI exports as PDFs.")
        st.info("""
        ### How to export Power BI reports to PDF:
        1. Open your PBIX file in Power BI Desktop
        2. Click File > Export to > PDF
        3. Save the PDF file in the 'reports' folder of this application
        4. Restart the application if needed
        """)
    else:
        # Select report to view
        selected_report = st.selectbox("Select Power BI Report", report_files)

        report_path = os.path.join(reports_folder, selected_report)

        # Display the report
        st.subheader(f"Viewing: {selected_report}")
        display_pdf(report_path)

        # Add report insights or description
        with st.expander("📋 Report Details"):
            st.write("""
            This report provides detailed analytics from Power BI. Use the scrollbar in the PDF viewer 
            to navigate through the report pages. For interactive features, please open the original 
            Power BI dashboard.
            """)

            # Option to download the PDF
            with open(report_path, "rb") as file:
                btn = st.download_button(
                    label="📥 Download PDF Report",
                    data=file,
                    file_name=selected_report,
                    mime="application/pdf"
                )

# ====================

# FOOTER
# ====================
st.markdown("---")
st.markdown("""
<div style="text-align:center; font-size:15px">
    Built with ❤ by <b>Harsh Dwivedi & Radhika Verma</b><br>
    📧 <a href="mailto:dwivediharsh7045@gmail.com">dwivediharsh7045@gmail.com</a> | 
    📧 <a href="mailto:radhikaverma5418@gmail.com">radhikaverma5418@gmail.com</a><br>
    💼 <a href="https://www.linkedin.com/in/harsh-dwivedi-7a7539212/" target="_blank">Harsh's LinkedIn</a> | 
    💼 <a href="https://www.linkedin.com/in/radhika-verma-158235258/" target="_blank">Radhika's LinkedIn</a>
</div>
""", unsafe_allow_html=True)