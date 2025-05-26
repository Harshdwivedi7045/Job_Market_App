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
import numpy as np
import spacy
from datetime import datetime
import plotly.graph_objects as go
import json
import time
import logging
import io

# Configure Logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Page Config
st.set_page_config(page_title="India Job Market Dashboard", layout="wide", initial_sidebar_state="expanded")

# Advanced CSS for Modern UI
st.markdown("""
    <style>
    /* General App Styling */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #e9ecef 0%, #ced4da 100%);
        transition: background 0.5s ease;
    }

    /* Sidebar Styling */
    .stSidebar {
        background: linear-gradient(180deg, #e0e6f0 0%, #d1d9e6 100%);
        color: #2c3e50;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    .stSidebar .stRadio > div, .stSidebar .stSelectbox, .stSidebar .stMultiSelect {
        background: #ffffff;
        border-radius: 8px;
        border: 1px solid #b0b8c4;
        padding: 8px;
    }
    .stSidebar .stRadio > div > label {
        color: #2c3e50;
    }

    /* Card Styling with Glassmorphism */
    .card {
        background: rgba(255, 255, 255, 0.7);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 20px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }

    /* Button Styling with Animation */
    .stButton>button {
        background: linear-gradient(90deg, #007bff 0%, #00d4ff 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        text-transform: uppercase;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0056b3 0%, #0098cc 100%);
        box-shadow: 0 4px 15px rgba(0,123,255,0.4);
        transform: scale(1.05);
    }
    .stButton>button::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255,255,255,0.3);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: width 0.4s ease, height 0.4s ease;
    }
    .stButton>button:hover::after {
        width: 200px;
        height: 200px;
    }

    /* Input Fields */
    .stSelectbox, .stMultiselect, .stTextInput, .stTextArea {
        background: #ffffff;
        border-radius: 8px;
        border: 1px solid #b0b8c4;
        transition: border-color 0.3s ease;
    }
    .stSelectbox:hover, .stMultiselect:hover, .stTextInput:hover, .stTextArea:hover {
        border-color: #007bff;
    }

    /* Tabs Styling */
    .stTabs {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }

    /* Plotly Chart Styling */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }

    /* Insight Box */
    .insight-box {
        background: rgba(227, 242, 253, 0.9);
        padding: 15px;
        border-left: 5px solid #2196f3;
        border-radius: 5px;
        margin: 10px 0;
        transition: background 0.3s ease;
    }
    .insight-box:hover {
        background: rgba(200, 230, 255, 0.9);
    }

    /* Notification Styling */
    .notification {
        background: rgba(40, 167, 69, 0.9);
        color: white;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        animation: slideIn 0.5s ease;
    }
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# Load Lottie Animation
@st.cache_data
def load_lottie_url(url):
    try:
        response = requests.get(url)
        if response.status_code != 200:
            logging.warning(f"Failed to load Lottie animation from {url}")
            return None
        return response.json()
    except Exception as e:
        logging.error(f"Error loading Lottie animation: {str(e)}")
        return None

# Load animations
animations = {
    "job": load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_touohxv0.json"),
    "company": load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_w51pcehl.json"),
    "skills": load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_jtbfg2nb.json"),
    "ml": load_lottie_url("https://assets8.lottiefiles.com/packages/lf20_UJNc2t.json"),
    "chatbot": load_lottie_url("https://assets8.lottiefiles.com/packages/lf20_7cmxb4dh.json"),
}

# Language Toggle
language = st.sidebar.selectbox("🌐 Language", ["English", "Hindi"])
lang_dict = {
    "English": {
        "title": "India Job Market Dashboard",
        "filter_jobs": "Filter Jobs",
        "skills_required": "Skills Required",
        "job_location": "Job Location",
        "experience_required": "Experience Required",
        "no_data": "No data found for the selected filters.",
        "download_csv": "Download CSV",
        "submit_feedback": "Submit Feedback",
        "feedback_comments": "Please provide suggestions before submitting.",
        "feedback_success": "Thank you for your feedback!",
        "feedback_error": "Failed to submit feedback. Please try again.",
        "save_profile": "Save Profile",
        "profile_updated": "Profile updated successfully!",
        "profile_error": "Failed to update profile. Please try again."
    },
    "Hindi": {
        "title": "भारत नौकरी बाजार डैशबोर्ड",
        "filter_jobs": "नौकरियां फ़िल्टर करें",
        "skills_required": "आवश्यक कौशल",
        "job_location": "नौकरी का स्थान",
        "experience_required": "आवश्यक अनुभव",
        "no_data": "चयनित फ़िल्टर के लिए कोई डेटा नहीं मिला।",
        "download_csv": "CSV डाउनलोड करें",
        "submit_feedback": "प्रतिक्रिया सबमिट करें",
        "feedback_comments": "कृपया सबमिट करने से पहले सुझाव प्रदान करें।",
        "feedback_success": "आपकी प्रतिक्रिया के लिए धन्यवाद!",
        "feedback_error": "प्रतिक्रिया सबमिट करने में विफल। कृपया पुनः प्रयास करें।",
        "save_profile": "प्रोफाइल सहेजें",
        "profile_updated": "प्रोफाइल सफलतापूर्वक अपडेट की गई!",
        "profile_error": "प्रोफाइल अपडेट करने में विफल। कृपया पुनः प्रयास करें।"
    }
}
lang = lang_dict[language]

# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/india_job_market_dataset.csv")
        logging.info("Dataset loaded successfully")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        st.error(f"Failed to load dataset: {str(e)}")
        return pd.DataFrame()

df = load_data()

# Sidebar Navigation with Icons
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "🏢 Company Insights",
    "💡 Skill Insights",
    "🤖 ML Analysis",
    "📈 Forecasting",
    "📊 Power BI Reports",
    "🤖 Chatbot Assistant",
    "📋 Job Application Tracker",
    "🎓 Skill Gap Analysis",
    "📝 Feedback Submission",
    "👤 User Profile"
], label_visibility="collapsed")

# Sidebar Filters
st.sidebar.header(lang["filter_jobs"])
selected_skills = st.sidebar.multiselect(lang["skills_required"], sorted(df['Skills Required'].dropna().unique()))
selected_city = st.sidebar.multiselect(lang["job_location"], sorted(df['Job Location'].dropna().unique()))
selected_experience = st.sidebar.multiselect(lang["experience_required"],
                                            sorted(df['Experience Required'].dropna().unique()))

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

# Real-Time Notification System
def check_new_jobs():
    try:
        new_jobs = filtered_df[filtered_df['Posted Date'].apply(
            lambda x: pd.to_datetime(x) > pd.Timestamp.now() - pd.Timedelta(days=1)
        )]
        if not new_jobs.empty:
            st.markdown(f"<div class='notification'>🔔 {len(new_jobs)} new jobs match your filters!</div>",
                        unsafe_allow_html=True)
            logging.info(f"New jobs detected: {len(new_jobs)}")
    except Exception as e:
        logging.error(f"Error in notification system: {str(e)}")

check_new_jobs()

# Load SpaCy Model with Error Handling
try:
    nlp = spacy.load("en_core_web_sm")
    logging.info("SpaCy model 'en_core_web_sm' loaded successfully")
except Exception as e:
    logging.error(f"Failed to load SpaCy model: {str(e)}")
    st.error(f"Failed to load SpaCy model: {str(e)}. Please ensure the 'en_core_web_sm' model is installed by running 'python -m spacy download en_core_web_sm'.")
    nlp = None

# Enhanced Chatbot with NLP
def get_chatbot_response(user_input):
    if nlp is None:
        return "Chatbot is disabled due to missing SpaCy model. Please install 'en_core_web_sm'."
    try:
        user_input = user_input.lower().strip()
        if not user_input:
            return "Please enter a question to get a response."

        logging.info(f"Processing chatbot input: {user_input}")
        doc = nlp(user_input)

        responses = {
            "top_skills": {
                "keywords": ["top skills", "in-demand skills", "popular skills", "best skills"],
                "response": "Based on the dataset, top in-demand skills include Python, Java, SQL, AWS, and Machine Learning. Check the Skill Insights page for a detailed analysis!"
            },
            "top_companies": {
                "keywords": ["top companies", "hiring companies", "best companies", "major employers"],
                "response": "Top hiring companies include TCS, Infosys, and Wipro. Visit the Company Insights page for a bar chart of top companies."
            },
            "job_locations": {
                "keywords": ["job locations", "popular cities", "job cities", "where are jobs"],
                "response": "Popular job locations are Bangalore, Hyderabad, Mumbai, and Delhi. The Company Insights page shows a pie chart of job distributions by city."
            },
            "how_to_use": {
                "keywords": ["how to use", "use dashboard", "navigate dashboard", "dashboard guide"],
                "response": "Use the sidebar to navigate pages, apply filters for skills, locations, or experience, and explore visualizations. Try the ML Analysis page for predictive insights!"
            },
            "salary_prediction": {
                "keywords": ["salary prediction", "predict salary", "salary range", "how much can i earn"],
                "response": "Go to the ML Analysis page and select Salary Prediction to input job details and get a predicted salary range."
            },
            "dashboard_info": {
                "keywords": ["what is this dashboard", "dashboard purpose", "about dashboard", "dashboard info",
                             "who built this dashboard", "who created this"],
                "response": "This is an India Job Market Dashboard built by Harsh Dwivedi and Radhika Verma. It analyzes job postings, skills, companies, and more using data visualizations and ML models."
            },
            "highest_paying_jobs": {
                "keywords": ["highest paying job roles", "high paying jobs", "best paying roles", "top salary jobs"],
                "response": "High-paying roles include Data Scientist, Machine Learning Engineer, and Software Architect, often offering 12-20 LPA or more. Check the ML Analysis page for salary predictions."
            },
            "most_job_opportunities": {
                "keywords": ["most job opportunities", "cities with most jobs", "job opportunities",
                             "where are most jobs"],
                "response": "Bangalore, Hyderabad, and Mumbai have the most job opportunities. Explore the Company Insights page for a pie chart of job distributions by city."
            },
            "improve_prospects": {
                "keywords": ["improve job prospects", "better job", "job chances", "career prospects"],
                "response": "Learn in-demand skills like Python, AWS, or Data Science, and gain 2-5 years of experience. Use the Skill Insights and ML Analysis pages to identify trends and network on LinkedIn."
            },
            "job_trend": {
                "keywords": ["job market trend", "trend for", "skill trend", "market trend"],
                "response": "To see trends for a specific skill, go to the Forecasting page, select Skill-wise forecasting, and choose your skill for a 90-day job posting forecast."
            },
            "skills_to_learn": {
                "keywords": ["skills to learn", "learn skills", "job skills", "skills for jobs",
                             "skills should i learn"],
                "response": "Focus on Python, SQL, AWS, Java, and Machine Learning for technical roles, plus soft skills like communication. Visit the Skill Insights page for top skills."
            },
            "fresher_jobs": {
                "keywords": ["hiring freshers", "fresher jobs", "entry level jobs", "jobs for freshers"],
                "response": "Companies like TCS, Infosys, and Wipro hire freshers (0-2 years experience). Filter by 0-2 years in the sidebar and check Company Insights for top companies."
            },
            "ml_accuracy": {
                "keywords": ["ml predictions accurate", "prediction accuracy", "how accurate", "ml reliability"],
                "response": "ML predictions like salary ranges are mock models for demonstration. Real accuracy depends on data and training. See ML Analysis for confidence scores."
            }
        }

        for key, value in responses.items():
            for keyword in value["keywords"]:
                if keyword in user_input:
                    logging.info(f"Matched keyword: {keyword} for response: {key}")
                    return value["response"]

        skills = set()
        for skills_list in df['Skills Required'].dropna():
            skills.update(skill.strip().lower() for skill in skills_list.split(','))
        for token in doc:
            if token.text in skills:
                logging.info(f"Matched skill: {token.text}")
                return "Trends for " + token.text.capitalize() + ": Check the Forecasting page for a 90-day forecast."

        try:
            with open("unrecognized_inputs.csv", "a", encoding='utf-8') as f:
                f.write(str(pd.Timestamp.now()) + "," + user_input + "\n")
            logging.info(f"Unrecognized input logged: {user_input}")
        except Exception as e:
            logging.error(f"Error logging unrecognized input: {str(e)}")

        return "I'm not sure how to answer that. Try asking about skills, companies, or locations!"
    except Exception as e:
        logging.error(f"Error in chatbot: {str(e)}")
        return f"Error: {str(e)}. Please try again."

# Generate Summary Report
def generate_summary_report():
    try:
        summary = {
            "Total Jobs": len(filtered_df),
            "Top Skills": Counter([skill.strip() for skills in filtered_df['Skills Required'].dropna() for skill in
                                   skills.split(',')]).most_common(5),
            "Top Cities": filtered_df['Job Location'].value_counts().head(5).to_dict(),
            "Top Companies": filtered_df['Company Name'].value_counts().head(5).to_dict()
        }
        summary_text = """
        # Job Market Summary Report
        *Generated on*: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """

        ## Overview
        - *Total Job Postings*: """ + str(summary['Total Jobs']) + """

        ## Top 5 In-Demand Skills
        """ + ''.join(["- " + skill + ": " + str(count) + " postings\n" for skill, count in summary['Top Skills']]) + """

        ## Top 5 Job Locations
        """ + ''.join(
            ["- " + city + ": " + str(count) + " postings\n" for city, count in summary['Top Cities'].items()]) + """

        ## Top 5 Hiring Companies
        """ + ''.join(
            ["- " + company + ": " + str(count) + " postings\n" for company, count in summary['Top Companies'].items()])
        return summary_text
    except Exception as e:
        logging.error(f"Error generating summary report: {str(e)}")
        return None

# ====================
# PAGE 1: HOME
# ====================
if page == "🏠 Home":
    st.title(lang["title"])
    if animations["job"]:
        st_lottie(animations["job"], height=250)
    st.markdown(
        "<div class='card'>Welcome to the enhanced " + lang[
            'title'] + "! Explore job trends, save applications, and provide feedback.</div>",
        unsafe_allow_html=True)

    st.subheader("🗂 Full Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("📊 Filtered Job Results")
    if filtered_df.empty:
        st.warning(lang["no_data"])
    else:
        st.dataframe(filtered_df, use_container_width=True)
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(lang["download_csv"], data=csv, file_name='filtered_job_data.csv', mime='text/csv')

    st.subheader("🗺 Job Locations Map")
    try:
        city_counts = filtered_df['Job Location'].value_counts().reset_index()
        city_counts.columns = ['City', 'Count']
        city_coords = {
            'Bangalore': [12.9716, 77.5946], 'Hyderabad': [17.3850, 78.4867], 'Mumbai': [19.0760, 72.8777],
            'Delhi': [28.7041, 77.1025], 'Chennai': [13.0827, 80.2707], 'Pune': [18.5204, 73.8567]
        }
        city_counts['lat'] = city_counts['City'].map(lambda x: city_coords.get(x, [0, 0])[0])
        city_counts['lon'] = city_counts['City'].map(lambda x: city_coords.get(x, [0, 0])[1])
        fig = px.scatter_mapbox(city_counts, lat="lat", lon="lon", size="Count", hover_name="City",
                                color="Count", zoom=3, height=400)
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
        map_html = fig.to_html()
        st.download_button("📥 Download Map as HTML", data=map_html, file_name="job_locations_map.html",
                           mime="text/html")
        st.markdown(
            "<div class='insight-box'>*Insight*: Larger dots indicate higher job concentrations in cities like Bangalore and Mumbai.</div>",
            unsafe_allow_html=True)
    except Exception as e:
        logging.error(f"Error rendering map: {str(e)}")
        st.error(f"Failed to render map: {str(e)}")

# ====================
# PAGE 2: COMPANY INSIGHTS
# ====================
elif page == "🏢 Company Insights":
    st.title("🏢 Company-Level Insights")
    if animations["company"]:
        st_lottie(animations["company"], height=250)

    st.subheader("🔝 Top Companies & Job Titles")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏆 Top Hiring Companies")
        company_chart = filtered_df['Company Name'].value_counts().head(10).reset_index()
        company_chart.columns = ['Company', 'Postings']
        fig1 = px.bar(company_chart, x='Company', y='Postings', color='Postings', color_continuous_scale='Viridis')
        st.plotly_chart(fig1, use_container_width=True)
        fig1_html = fig1.to_html()
        st.download_button("📥 Download Companies Chart", data=fig1_html, file_name="top_companies.html",
                           mime="text/html")
        st.markdown(
            "<div class='insight-box'>*Insight*: Leading IT firms dominate hiring, reflecting strong demand in tech sectors.</div>",
            unsafe_allow_html=True)

        st.markdown("#### 💼 Popular Job Titles")
        title_chart = filtered_df['Job Title'].value_counts().head(10).reset_index()
        title_chart.columns = ['Title', 'Count']
        fig2 = px.bar(title_chart, x='Title', y='Count', color='Count', color_continuous_scale='Plasma')
        st.plotly_chart(fig2, use_container_width=True)
        fig2_html = fig2.to_html()
        st.download_button("📥 Download Job Titles Chart", data=fig2_html, file_name="job_titles.html", mime="text/html")

    with col2:
        st.markdown("#### 📍 Jobs by City")
        city_chart = filtered_df['Job Location'].value_counts().head(10).reset_index()
        city_chart.columns = ['City', 'Count']
        fig3 = px.pie(city_chart, names='City', values='Count', color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig3, use_container_width=True)
        fig3_html = fig3.to_html()
        st.download_button("📥 Download City Chart", data=fig3_html, file_name="jobs_by_city.html", mime="text/html")

        st.markdown("#### 🎯 Experience Demand")
        experience_chart = filtered_df['Experience Required'].value_counts().reset_index()
        experience_chart.columns = ['Experience', 'Count']
        fig4 = px.line(experience_chart.sort_values('Experience'), x='Experience', y='Count', markers=True,
                       color_discrete_sequence=['#FF5722'])
        st.plotly_chart(fig4, use_container_width=True)
        fig4_html = fig4.to_html()
        st.download_button("📥 Download Experience Chart", data=fig4_html, file_name="experience_demand.html",
                           mime="text/html")

    st.subheader("🧾 Job Type Distribution")
    job_type_chart = filtered_df['Job Type'].value_counts().reset_index()
    job_type_chart.columns = ['Job Type', 'Count']
    fig5 = px.pie(job_type_chart, names='Job Type', values='Count', color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig5, use_container_width=True)
    fig5_html = fig5.to_html()
    st.download_button("📥 Download Job Type Chart", data=fig5_html, file_name="job_types.html", mime="text/html")

# ====================
# PAGE 3: SKILL INSIGHTS
# ====================
elif page == "💡 Skill Insights":
    st.title("💡 In-Demand Skills Analysis")
    if animations["skills"]:
        st_lottie(animations["skills"], height=250)

    all_skills = []
    for row in filtered_df['Skills Required'].dropna():
        all_skills.extend([s.strip() for s in row.split(',')])

    if all_skills:
        top_skills = Counter(all_skills).most_common(10)
        skills_df = pd.DataFrame(top_skills, columns=['Skill', 'Count'])
        st.markdown("#### 🔝 Top 10 In-Demand Skills")
        fig = px.bar(skills_df, x='Skill', y='Count', color='Count', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
        fig_html = fig.to_html()
        st.download_button("📥 Download Skills Chart", data=fig_html, file_name="top_skills.html", mime="text/html")
        st.markdown(
            "<div class='insight-box'>*Insight*: Programming and cloud skills are highly sought after, indicating a tech-driven market.</div>",
            unsafe_allow_html=True)
    else:
        st.info("ℹ No skill data available for selected filters.")

# ====================
# PAGE 4: ML ANALYSIS
# ====================
elif page == "🤖 ML Analysis":
    st.title("🤖 Machine Learning Analysis")
    if animations["ml"]:
        st_lottie(animations["ml"], height=250)

    ml_tabs = st.tabs(["Salary Prediction", "Job Clustering", "Experience Impact", "Skill Analysis"])

    with ml_tabs[0]:
        st.header("💰 Salary Prediction Model")
        st.subheader("Enter Job Details")
        col1, col2 = st.columns(2)

        with col1:
            pred_job_title = st.selectbox("Job Title", sorted(df['Job Title'].unique()))
            pred_company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
            pred_job_type = st.selectbox("Job Type", sorted(df['Job Type'].unique()))
            pred_job_location = st.selectbox("Job Location", sorted(df['Job Location'].unique()))

        with col2:
            pred_experience = st.selectbox("Experience Required", sorted(df['Experience Required'].unique()))
            pred_education = st.selectbox("Education Level",
                                          ["Bachelor's", "Master's", "PhD", "MBA", "High School", "Diploma"])
            all_skills = set()
            for skills in df['Skills Required'].dropna():
                all_skills.update([s.strip() for s in skills.split(',')])
            pred_skills = st.multiselect("Skills", sorted(all_skills), max_selections=5)
            pred_remote = st.selectbox("Work Mode", ["Remote", "Hybrid", "Onsite"])

        if st.button("Predict Salary Range"):
            try:
                st.info("🧠 Calculating prediction...")
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                salary_brackets = ["3-5 LPA", "5-8 LPA", "8-12 LPA", "12-20 LPA", "20+ LPA"]
                confidence_scores = [np.random.uniform(0.1, 0.9) for _ in range(len(salary_brackets))]
                total = sum(confidence_scores)
                normalized_scores = [score / total for score in confidence_scores]
                st.success("✅ Prediction complete!")
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.subheader("Predicted Salary Range")
                    pred_df = pd.DataFrame({
                        'Salary Range': salary_brackets,
                        'Probability': normalized_scores
                    })
                    fig = px.bar(pred_df, x='Salary Range', y='Probability',
                                 color='Probability',
                                 labels={'Probability': 'Confidence'},
                                 color_continuous_scale=px.colors.sequential.Viridis)
                    top_prediction = salary_brackets[normalized_scores.index(max(normalized_scores))]
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(
                        "**Most likely salary range: **" + top_prediction + " (Confidence: " + f"{max(normalized_scores):.2%})")
                    fig_html = fig.to_html()
                    st.download_button("📥 Download Salary Chart", data=fig_html, file_name="salary_prediction.html",
                                       mime="text/html")
                with col2:
                    st.subheader("Key Factors")
                    factors = [
                        {"factor": "Experience",
                         "impact": "High 📈" if "5-10" in pred_experience or "10+" in pred_experience else "Medium ⚖"},
                        {"factor": "Education",
                         "impact": "High 📈" if pred_education in ["PhD", "MBA"] else "Medium ⚖"},
                        {"factor": "Location",
                         "impact": "High 📈" if pred_job_location in ["Bangalore", "Delhi", "Mumbai"] else "Medium ⚖"},
                        {"factor": "Job Type", "impact": "Medium ⚖" if pred_job_type == "Full-time" else "Low 📉"}
                    ]
                    for factor in factors:
                        st.markdown("" + factor['factor'] + ": " + factor['impact'])
                logging.info("Salary prediction completed successfully")
            except Exception as e:
                logging.error(f"Error in salary prediction: {str(e)}")
                st.error(f"Failed to predict salary: {str(e)}")

    with ml_tabs[1]:
        st.header("🔍 Job Market Clustering")
        st.subheader("Explore Job Market Segments")
        st.markdown("#### Configure Clustering Parameters")
        cluster_col1, cluster_col2 = st.columns(2)
        with cluster_col1:
            n_clusters = st.slider("Number of Clusters", min_value=2, max_value=10, value=5)
            clustering_algorithm = st.selectbox("Clustering Algorithm",
                                                ["K-Means", "Hierarchical", "DBSCAN"])
        with cluster_col2:
            feature_set = st.multiselect("Features for Clustering",
                                         ["Skills", "Experience", "Location", "Job Type", "Company Size"],
                                         default=["Skills", "Experience"])
            visualization_type = st.selectbox("Visualization Type", ["2D Plot", "3D Plot", "Dendrogram"])
        if st.button("Run Clustering Analysis"):
            try:
                st.info("🧮 Performing clustering analysis...")
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                n_samples = 1000
                np.random.seed(42)
                mock_features = np.random.rand(n_samples, 2)
                mock_clusters = np.random.randint(0, n_clusters, size=n_samples)
                cluster_df = pd.DataFrame({
                    'x': mock_features[:, 0],
                    'y': mock_features[:, 1],
                    'cluster': mock_clusters
                })
                st.success("✅ Clustering complete!")
                st.subheader("Clustering Results")
                viz_col1, viz_col2 = st.columns([2, 1])
                with viz_col1:
                    if visualization_type == "2D Plot":
                        fig = px.scatter(cluster_df, x='x', y='y', color='cluster',
                                         color_continuous_scale=px.colors.qualitative.G10,
                                         labels={'cluster': 'Job Cluster'},
                                         title="Job Market Clusters using " + clustering_algorithm)
                        st.plotly_chart(fig, use_container_width=True)
                        fig_html = fig.to_html()
                        st.download_button("📥 Download Cluster Plot", data=fig_html, file_name="job_clusters.html",
                                           mime="text/html")
                    elif visualization_type == "3D Plot":
                        cluster_df['z'] = np.random.rand(n_samples)
                        fig = px.scatter_3d(cluster_df, x='x', y='y', z='z', color='cluster',
                                            color_continuous_scale=px.colors.qualitative.G10)
                        st.plotly_chart(fig, use_container_width=True)
                        fig_html = fig.to_html()
                        st.download_button("📥 Download 3D Cluster Plot", data=fig_html,
                                           file_name="job_clusters_3d.html", mime="text/html")
                    else:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        from scipy.cluster import hierarchy
                        from scipy.spatial.distance import pdist

                        mock_data = np.random.rand(50, 2)
                        mock_linkage = hierarchy.linkage(pdist(mock_data), method='ward')
                        hierarchy.dendrogram(mock_linkage, ax=ax)
                        ax.set_title('Hierarchical Clustering Dendrogram')
                        st.pyplot(fig)
                with viz_col2:
                    st.markdown("### Cluster Interpretation")
                    for i in range(n_clusters):
                        with st.expander("Cluster " + str(i + 1) + " Characteristics"):
                            st.markdown("*Size*: " + str(np.sum(mock_clusters == i)) + " jobs")
                            if "Skills" in feature_set:
                                skill_groups = [
                                    "Python, SQL, Data Analysis",
                                    "Java, Spring Boot, Microservices",
                                    "Frontend: React, Angular, JavaScript",
                                    "ML/AI: TensorFlow, PyTorch, NLP",
                                    "Cloud: AWS, Azure, DevOps"
                                ]
                                st.markdown("*Key Skills*: " + skill_groups[i % len(skill_groups)])
                            if "Experience" in feature_set:
                                exp_groups = ["0-2 years", "3-5 years", "5-8 years", "8+ years"]
                                st.markdown("*Experience Level*: " + exp_groups[i % len(exp_groups)])
                            if "Location" in feature_set:
                                location_groups = ["Bangalore, Hyderabad", "Mumbai, Pune", "Delhi NCR",
                                                   "Chennai, Kolkata"]
                                st.markdown("*Common Locations*: " + location_groups[i % len(location_groups)])
                            salary_ranges = ["4-7 LPA", "8-12 LPA", "12-18 LPA", "18-25 LPA", "25+ LPA"]
                            st.markdown("*Salary Range*: " + salary_ranges[i % len(salary_ranges)])
                cluster_result_df = pd.DataFrame({
                    'Job Title': np.random.choice(df['Job Title'].unique(), size=n_samples),
                    'Cluster': mock_clusters,
                    'Similarity Score': np.random.uniform(0.6, 0.99, size=n_samples)
                })
                csv = cluster_result_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Cluster Results",
                                   data=csv,
                                   file_name="job_clusters.csv",
                                   mime="text/csv")
                logging.info("Clustering analysis completed successfully")
            except Exception as e:
                logging.error(f"Error in clustering analysis: {str(e)}")
                st.error(f"Failed to perform clustering: {str(e)}")

    with ml_tabs[2]:
        st.header("📈 Experience Impact Analysis")
        st.subheader("Analyze Experience Demand and Salary Impact")
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("#### Job Postings by Experience Level")
            exp_chart = filtered_df['Experience Required'].value_counts().reset_index()
            exp_chart.columns = ['Experience', 'Count']
            fig_exp = px.bar(exp_chart.sort_values('Experience'), x='Experience', y='Count',
                             color='Count', color_continuous_scale='Blues')
            st.plotly_chart(fig_exp, use_container_width=True)
            fig_exp_html = fig_exp.to_html()
            st.download_button("📥 Download Experience Chart", data=fig_exp_html,
                               file_name="experience_postings.html", mime="text/html")
            st.markdown(
                "<div class='insight-box'>*Insight*: Mid-level experience (3-8 years) often has the highest demand.</div>",
                unsafe_allow_html=True)

        with exp_col2:
            st.markdown("#### Mock Salary Impact by Experience")
            exp_input = st.selectbox("Select Experience Level", sorted(df['Experience Required'].unique()))
            if st.button("Estimate Salary Impact"):
                try:
                    st.info("🧠 Estimating salary impact...")
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    salary_ranges = ["3-5 LPA", "5-8 LPA", "8-12 LPA", "12-20 LPA", "20+ LPA"]
                    mock_salaries = {
                        "0-2 years": [0.7, 0.2, 0.05, 0.03, 0.02],
                        "3-5 years": [0.1, 0.6, 0.2, 0.08, 0.02],
                        "5-8 years": [0.05, 0.15, 0.5, 0.25, 0.05],
                        "8-12 years": [0.02, 0.05, 0.2, 0.5, 0.23],
                        "12+ years": [0.01, 0.03, 0.1, 0.36, 0.5]
                    }
                    probabilities = mock_salaries.get(exp_input, [0.2] * 5)
                    salary_df = pd.DataFrame({
                        'Salary Range': salary_ranges,
                        'Probability': probabilities
                    })
                    fig_salary = px.bar(salary_df, x='Salary Range', y='Probability',
                                        color='Probability', color_continuous_scale='Reds')
                    st.plotly_chart(fig_salary, use_container_width=True)
                    top_salary = salary_ranges[probabilities.index(max(probabilities))]
                    st.markdown(
                        "*Estimated Salary Range*: " + top_salary + " (Confidence: " + f"{max(probabilities):.2%})")
                    fig_salary_html = fig_salary.to_html()
                    st.download_button("📥 Download Salary Impact Chart", data=fig_salary_html,
                                       file_name="experience_salary_impact.html", mime="text/html")
                    st.success("✅ Estimation complete!")
                    logging.info(f"Experience salary impact estimated for: {exp_input}")
                except Exception as e:
                    logging.error(f"Error in experience salary impact: {str(e)}")
                    st.error(f"Failed to estimate salary impact: {str(e)}")

    with ml_tabs[3]:
        st.header("🔍 Skill Demand Analysis")
        st.subheader("Analyze Skills by Job Role or Location")
        skill_col1, skill_col2 = st.columns(2)

        with skill_col1:
            st.markdown("#### Select Analysis Type")
            analysis_type = st.radio("Analyze Skills By", ["Job Role", "Location"], horizontal=True)
            if analysis_type == "Job Role":
                selected_role = st.selectbox("Select Job Role", sorted(df['Job Title'].unique()))
            else:
                selected_location = st.selectbox("Select Location", sorted(df['Job Location'].unique()))

        with skill_col2:
            st.markdown("#### Top Skills Required")
            if st.button("Analyze Skills"):
                try:
                    st.info("🧠 Analyzing skill demand...")
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    if analysis_type == "Job Role":
                        skill_data = filtered_df[filtered_df['Job Title'] == selected_role]['Skills Required'].dropna()
                    else:
                        skill_data = filtered_df[filtered_df['Job Location'] == selected_location][
                            'Skills Required'].dropna()
                    all_skills = []
                    for skills in skill_data:
                        all_skills.extend([s.strip() for s in skills.split(',')])
                    if all_skills:
                        skill_counts = Counter(all_skills).most_common(10)
                        skill_df = pd.DataFrame(skill_counts, columns=['Skill', 'Count'])
                        fig_skill = px.bar(skill_df, x='Skill', y='Count',
                                           color='Count', color_continuous_scale='Greens')
                        st.plotly_chart(fig_skill, use_container_width=True)
                        fig_skill_html = fig_skill.to_html()
                        st.download_button("📥 Download Skill Demand Chart", data=fig_skill_html,
                                           file_name="skill_demand.html", mime="text/html")
                        st.markdown(
                            "<div class='insight-box'>*Insight*: Dominant skills reflect role or location-specific demands.</div>",
                            unsafe_allow_html=True)
                        st.success("✅ Analysis complete!")
                        logging.info(
                            f"Skill demand analyzed for {analysis_type}: {selected_role if analysis_type == 'Job Role' else selected_location}")
                    else:
                        st.warning("⚠ No skill data available for the selected criteria.")
                except Exception as e:
                    logging.error(f"Error in skill demand analysis: {str(e)}")
                    st.error(f"Failed to analyze skills: {str(e)}")

# ====================
# PAGE 5: FORECASTING
# ====================
elif page == "📈 Forecasting":
    st.title("📈 Forecasting Job Postings")
    if animations["job"]:
        st_lottie(animations["job"], height=250)

    if 'Posted Date' not in df.columns:
        st.error("❌ 'Posted Date' column not found in dataset.")
    else:
        try:
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
                model = Prophet()
                model.fit(ts_df)
                future = model.make_future_dataframe(periods=90)
                forecast = model.predict(future)
                st.subheader("📊 Forecasted Job Postings (Next 90 Days)")
                fig = plot_plotly(model, forecast)
                st.plotly_chart(fig, use_container_width=True)
                fig_html = fig.to_html()
                st.download_button("📥 Download Forecast Chart", data=fig_html, file_name="job_forecast.html",
                                   mime="text/html")
                forecast_csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(
                    columns={'ds': 'Date', 'yhat': 'Forecasted Count'}
                )
                csv = forecast_csv.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Forecast CSV", data=csv, file_name="forecast_data.csv", mime="text/csv")
                with st.expander("📉 Trend & Seasonality Breakdown"):
                    st.pyplot(model.plot_components(forecast))
                st.markdown(
                    "<div class='insight-box'>*Insight*: Forecast trends help identify upcoming hiring surges.</div>",
                    unsafe_allow_html=True)
                logging.info("Forecasting completed successfully")
        except Exception as e:
            logging.error(f"Error in forecasting: {str(e)}")
            st.error(f"Failed to generate forecast: {str(e)}")

# ====================
# PAGE 6: POWER BI REPORTS
# ====================
elif page == "📊 Power BI Reports":
    st.title("📊 Power BI Analytics Reports")
    reports_folder = "reports"
    try:
        if not os.path.exists(reports_folder):
            os.makedirs(reports_folder)
            st.warning("⚠ 'reports' folder created. Please add your Power BI PDF exports.")
        report_files = [f for f in os.listdir(reports_folder) if f.endswith('.pdf')]
        if not report_files:
            st.error("❌ No PDF reports found in the 'reports' folder. Please add your Power BI exports.")
            st.info("""
            ### How to export Power BI reports to PDF:
            1. Open your PBIX file in Power BI Desktop
            2. Click File > Export to > PDF
            3. Save the PDF file in the 'reports' folder
            4. Restart the application if needed
            """)
        else:
            selected_report = st.selectbox("Select Power BI Report", report_files)
            report_path = os.path.join(reports_folder, selected_report)
            st.subheader("Viewing: " + selected_report)
            with open(report_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = """
                <iframe src="data:application/pdf;base64,""" + base64_pdf + """" width="100%" height="600" type="application/pdf"></iframe>
            """
            st.markdown(pdf_display, unsafe_allow_html=True)
            with st.expander("📋 Report Details"):
                st.write("""
                This report provides detailed analytics from Power BI. Use the scrollbar to navigate pages.
                For interactive features, open the original Power BI dashboard.
                """)
                with open(report_path, "rb") as file:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=file,
                        file_name=selected_report,
                        mime="application/pdf"
                    )
            logging.info(f"Displayed Power BI report: {selected_report}")
    except Exception as e:
        logging.error(f"Error in Power BI reports: {str(e)}")
        st.error(f"Failed to load Power BI reports: {str(e)}")

# ====================
# PAGE 7: CHATBOT ASSISTANT
# ====================
elif page == "🤖 Chatbot Assistant":
    st.title("🤖 Job Market Chatbot Assistant")
    if animations["chatbot"]:
        st_lottie(animations["chatbot"], height=250)

    st.markdown("""
    <div class='card'>
    Welcome to the Job Market Chatbot! Ask about:
    - Top skills or companies
    - Job locations or salary predictions
    - How to use the dashboard
    - Highest paying job roles
    - Cities with most job opportunities
    - Improving job prospects
    - Job market trends for specific skills
    </div>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me anything about the job market or dashboard..."):
        try:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_chatbot_response(prompt)
                    if response:
                        st.markdown(response)
                    else:
                        st.error("No response generated. Please try rephrasing.")
            st.session_state.chat_history.append({"role": "assistant", "content": response or "No response generated."})
            logging.info(f"Chatbot responded to: {prompt}")
        except Exception as e:
            logging.error(f"Error in chatbot response: {str(e)}")
            st.error(f"Error generating response: {str(e)}")
            st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})

# ====================
# PAGE 8: JOB APPLICATION TRACKER
# ====================
elif page == "📋 Job Application Tracker":
    st.title("📋 Job Application Tracker")
    st.markdown("<div class='card'>Save and track your job applications with detailed statuses.</div>",
                unsafe_allow_html=True)

    if "saved_jobs" not in st.session_state:
        st.session_state.saved_jobs = []

    st.subheader("Save a Job")
    job_index = st.selectbox("Select Job to Save", filtered_df.index, format_func=lambda
        x: filtered_df.loc[x, 'Job Title'] + " at " + filtered_df.loc[x, 'Company Name'])
    status = st.selectbox("Application Status", ["Not Applied", "Applied", "Interview", "Offer", "Rejected"])
    notes = st.text_area("Notes", placeholder="Add any notes about this application...")
    if st.button("Save Job"):
        try:
            job_data = filtered_df.loc[job_index].to_dict()
            job_data.update({
                "Status": status,
                "Notes": notes,
                "Saved Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.session_state.saved_jobs.append(job_data)
            st.success("✅ Job saved successfully!")
            logging.info(f"Job saved: {job_data['Job Title']} at {job_data['Company Name']}")
        except Exception as e:
            logging.error(f"Error saving job: {str(e)}")
            st.error(f"Failed to save job: {str(e)}")

    st.subheader("Your Saved Jobs")
    if st.session_state.saved_jobs:
        saved_df = pd.DataFrame(st.session_state.saved_jobs)
        st.dataframe(saved_df[['Job Title', 'Company Name', 'Job Location', 'Status', 'Saved Date', 'Notes']],
                     use_container_width=True)
        csv = saved_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Saved Jobs", data=csv, file_name="saved_jobs.csv", mime="text/csv")
    else:
        st.info("ℹ No jobs saved yet.")

# ====================
# PAGE 9: SKILL GAP ANALYSIS
# ====================
elif page == "🎓 Skill Gap Analysis":
    st.title("🎓 Skill Gap Analysis")
    st.markdown("<div class='card'>Identify gaps between your skills and job market demands.</div>",
                unsafe_allow_html=True)

    st.subheader("Your Skills")
    user_skills = st.multiselect("Select Your Skills", sorted(
        {skill.strip() for skills in df['Skills Required'].dropna() for skill in skills.split(',')}))

    st.subheader("Target Job Role")
    target_role = st.selectbox("Select Target Job Role", sorted(df['Job Title'].unique()))

    if st.button("Analyze Skill Gap"):
        try:
            role_skills = set()
            for skills in df[df['Job Title'] == target_role]['Skills Required'].dropna():
                role_skills.update([s.strip().lower() for s in skills.split(',')])
            user_skills_set = set([s.lower() for s in user_skills])
            missing_skills = role_skills - user_skills_set
            matched_skills = role_skills & user_skills_set

            st.subheader("Analysis Results")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### ✅ Matched Skills")
                if matched_skills:
                    for skill in matched_skills:
                        st.markdown("- " + skill.capitalize())
                else:
                    st.info("No matching skills found.")
            with col2:
                st.markdown("### ❗ Missing Skills")
                if missing_skills:
                    for skill in missing_skills:
                        st.markdown("- " + skill.capitalize())
                else:
                    st.success("You have all required skills!")

            if missing_skills:
                st.markdown("### 📚 Learning Recommendations")
                for skill in missing_skills:
                    with st.expander("Learn " + skill.capitalize()):
                        st.write("*Resources for " + skill.capitalize() + ":*")
                        st.markdown("- Online courses (e.g., Coursera, Udemy)")
                        st.markdown("- Official documentation or books")
                        st.markdown("- Practice projects or certifications")
            logging.info(f"Skill gap analysis completed for role: {target_role}")
        except Exception as e:
            logging.error(f"Error in skill gap analysis: {str(e)}")
            st.error(f"Failed to analyze skill gap: {str(e)}")

# ====================
# PAGE 10: FEEDBACK SUBMISSION
# ====================
elif page == "📝 Feedback Submission":
    st.title("📝 Feedback Submission")
    st.markdown("<div class='card'>Help us improve the dashboard by sharing your feedback!</div>",
                unsafe_allow_html=True)

    feedback = st.text_area("Your Feedback", placeholder="Share your suggestions or issues...")
    rating = st.slider("Rate the Dashboard (1-5)", 1, 5, 3)
    if st.button(lang["submit_feedback"]):
        if not feedback.strip():
            st.warning(lang["feedback_comments"])
        else:
            try:
                with open("feedback.csv", "a", encoding='utf-8') as f:
                    f.write(str(pd.Timestamp.now()) + "," + str(rating) + "," + feedback.replace(',', ' ') + "\n")
                st.success(lang["feedback_success"])
                logging.info(f"Feedback submitted: Rating {rating}, Comments: {feedback[:50]}...")
            except Exception as e:
                logging.error(f"Error submitting feedback: {str(e)}")
                st.error(lang["feedback_error"])

# ====================
# PAGE 11: USER PROFILE
# ====================
elif page == "👤 User Profile":
    st.title("👤 User Profile")
    st.markdown("<div class='card'>Manage your profile and preferences.</div>", unsafe_allow_html=True)

    if "user_profile" not in st.session_state:
        st.session_state.user_profile = {
            "name": "",
            "email": "",
            "preferred_city": "",
            "preferred_skills": []
        }

    st.subheader("Profile Details")
    name = st.text_input("Name", value=st.session_state.user_profile["name"])
    email = st.text_input("Email", value=st.session_state.user_profile["email"])
    preferred_city = st.selectbox("Preferred Job Location", sorted(df['Job Location'].dropna().unique()),
                                  index=df['Job Location'].dropna().unique().tolist().index(
                                      st.session_state.user_profile["preferred_city"])
                                  if st.session_state.user_profile["preferred_city"] in df[
                                      'Job Location'].dropna().unique() else 0)
    preferred_skills = st.multiselect("Preferred Skills", sorted(
        {skill.strip() for skills in df['Skills Required'].dropna() for skill in skills.split(',')}),
                                      default=st.session_state.user_profile["preferred_skills"])

    if st.button(lang["save_profile"]):
        try:
            st.session_state.user_profile = {
                "name": name,
                "email": email,
                "preferred_city": preferred_city,
                "preferred_skills": preferred_skills
            }
            with open("user_profile.json", "w", encoding='utf-8') as f:
                json.dump(st.session_state.user_profile, f)
            st.success(lang["profile_updated"])
            logging.info(f"User profile updated: {name}, {email}")
        except Exception as e:
            logging.error(f"Error saving profile: {str(e)}")
            st.error(lang["profile_error"])

    st.subheader("Generate Summary Report")
    if st.button("Download Summary Report"):
        try:
            report_text = generate_summary_report()
            if report_text:
                st.markdown(report_text)
                buffer = io.StringIO()
                buffer.write(report_text)
                buffer.seek(0)
                st.download_button(
                    label="📥 Download Report as TXT",
                    data=buffer.getvalue(),
                    file_name="job_market_summary.txt",
                    mime="text/plain"
                )
                logging.info("Summary report downloaded")
            else:
                st.error("Failed to generate summary report.")
        except Exception as e:
            logging.error(f"Error downloading summary report: {str(e)}")
            st.error(f"Failed to download report: {str(e)}")