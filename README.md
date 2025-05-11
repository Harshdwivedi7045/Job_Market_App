# India Job Market Dashboard

![Dashboard Preview](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Prophet](https://img.shields.io/badge/Prophet-Forecasting-blue?style=for-the-badge)
![Plotly](https://img.shields.io/badge/Plotly-Visualizations-3F4F75?style=for-the-badge&logo=plotly)

A comprehensive dashboard for analyzing the Indian job market, featuring interactive visualizations, machine learning insights, and future job trend forecasting.

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data](#-data)
- [Pages & Functionality](#-pages--functionality)
- [ML Models](#-ml-models)
- [Power BI Integration](#-power-bi-integration)
- [Authors](#-authors)
- [License](#-license)

## ✨ Features

- **Interactive filtering** by skills, location, and experience
- **Company insights** with visual representations of job market trends
- **In-demand skills analysis** to identify most sought-after competencies
- **ML-powered analysis** including salary predictions and job clustering
- **Job trend forecasting** using Prophet for time series forecasting
- **Power BI report integration** for additional custom analytics
- **Responsive UI** with light/dark mode toggle

## 🎥 Demo

*Add screenshots or GIF of your dashboard here*

## 🚀 Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/india-job-market-dashboard.git
cd india-job-market-dashboard

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## 📝 Requirements

```
streamlit
pandas
plotly
prophet
streamlit-lottie
requests
matplotlib
pillow
```

## 🛠 Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501` to view the dashboard.

## 📊 Data

The dashboard uses a dataset of Indian job market data named `india_job_market_dataset.csv`. The dataset should include:

- Job titles and descriptions
- Company information
- Required skills
- Experience requirements
- Job locations
- Posting dates
- Salary ranges (if available)

Place your dataset in a `data` folder at the root of the project.

## 📑 Pages & Functionality

### 1. Home
- Overview of the dataset
- Data preview and filtering options
- Download filtered job data as CSV

### 2. Company Insights
- Top hiring companies visualization
- Popular job titles analysis
- Geographic distribution of jobs
- Experience demand trends
- Job type distribution

### 3. Skill Insights
- Analysis of in-demand skills
- Skill frequency visualizations
- Skill relationships

### 4. ML Analysis
- Salary prediction model
- Job clustering analysis
- Skill demand prediction
- Experience impact assessment

### 5. Forecasting
- Job posting trend forecasting
- City-wise forecasting
- Skill-wise demand forecasting
- Trend and seasonality breakdown

### 6. Power BI Reports
- Integration with Power BI exported reports
- PDF viewer for report visualization
- Option to download reports

## 🤖 ML Models

The dashboard includes several machine learning features:

1. **Salary Prediction**: Predicts salary ranges based on job characteristics
2. **Job Clustering**: Groups similar jobs together
3. **Skill Demand Analysis**: Analyzes which skills are increasing in demand
4. **Experience Impact**: Measures how experience affects job prospects

## 📈 Power BI Integration

To use the Power BI integration feature:

1. Export your Power BI dashboards as PDFs
2. Place them in the `reports` folder
3. Access them through the "Power BI Reports" page

## 👨‍💻 Authors

- **Harsh Dwivedi** - [LinkedIn](https://www.linkedin.com/in/harsh-dwivedi-7a7539212/) - [Email](mailto:dwivediharsh7045@gmail.com)
- **Radhika Verma** - [LinkedIn](https://www.linkedin.com/in/radhika-verma-158235258/) - [Email](mailto:radhikaverma5418@gmail.com)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Built with ❤️ by Harsh Dwivedi & Radhika Verma
