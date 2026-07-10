# 🚀 Customer Churn Prediction System

An end-to-end Machine Learning web application that predicts customer churn using customer information. The project includes model explainability using SHAP, an interactive Streamlit dashboard, Docker support, and automated CI with GitHub Actions.

![Python](https://img.shields.io/badge/Python-3.11-blue)

![Streamlit](https://img.shields.io/badge/Streamlit-App-red)

![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)

![Docker](https://img.shields.io/badge/Docker-Container-blue)

![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI-success)

## 🌐 Live Demo

👉 https://customer-churn-prediction-hashini.streamlit.app/

## ✨ Features

- Predict customer churn using Machine Learning
- Interactive Streamlit web application
- SHAP Explainable AI visualizations
- Customer analytics dashboard
- Prediction history tracking
- Responsive user interface
- Docker container support
- Automated GitHub Actions workflow

## 🛠 Tech Stack

| Category | Technologies |
|----------|--------------|
| Language | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, SHAP |
| Web Framework | Streamlit |
| Deployment | Streamlit Cloud |
| Containerization | Docker |
| Version Control | Git & GitHub |
| CI/CD | GitHub Actions |

## 📂 Project Structure

Customer_Churn_Prediction/

├── app/

├── data/

├── models/

├── notebooks/

├── reports/

├── src/

├── Dockerfile

├── requirements.txt

└── README.md

## 📷 Screenshots 

### Home Page

![Home](screenshots/home.png)

---

### Prediction Result

![Prediction](screenshots/prediction.png)

---

### SHAP Explainability

![SHAP](screenshots/shap.png)

---

### Analytics Dashboard

![Dashboard](images/dashboard.png)

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Hashini-gitch/Customer_Churn_Prediction.git
```

Move into the project

```bash
cd Customer_Churn_Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app/app_v2.py
```

## 🐳 Docker

Build

```bash
docker build -t customer-churn-app .
```

Run

```bash
docker run -p 8501:8501 customer-churn-app
```

## 🚀 Future Improvements

- Cloud model monitoring
- Database integration
- REST API using FastAPI
- User authentication
- Multiple machine learning models
- MLOps pipeline

## 👩‍💻 Author

**Hashini Avishka Rathnayake**

Third-Year Data Science Undergraduate

Sri Lanka Institute of Information Technology (SLIIT)

GitHub:
https://github.com/Hashini-gitch

LinkedIn:
https://www.linkedin.com/in/hashini-a-rathnayake-2a8ba235b

## 📜 License

This project is developed for educational and portfolio purposes.
