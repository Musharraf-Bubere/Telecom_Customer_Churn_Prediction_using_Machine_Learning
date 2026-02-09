# 📊 Telecom Customer Churn Prediction using Machine Learning

An end-to-end Machine Learning project that predicts customer churn in the telecom industry using EDA, feature engineering, multiple ML models, and Streamlit deployment.

---

## 🧠 Project Overview

Customer churn is one of the biggest challenges in the telecom industry. Retaining existing customers is more cost-effective than acquiring new ones.

This project focuses on:
- Understanding customer behavior through Exploratory Data Analysis (EDA)
- Building machine learning classification models
- Predicting whether a customer is likely to churn
- Deploying the trained model using Streamlit

The goal is to help telecom companies take proactive actions to reduce customer churn.

---

## 🎯 Objectives

- Analyze telecom customer data
- Identify churn-driving factors
- Build accurate ML models
- Compare multiple algorithms
- Provide real-time churn prediction using a web app

---

## 📂 Dataset Description

The dataset contains customer-level information including:
- Customer demographics
- Services subscribed (Internet, Phone, etc.)
- Contract type
- Billing and payment details
- Monthly and total charges
- Target variable: Churn (Yes / No)

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed to:
- Understand data distribution
- Detect missing values
- Analyze correlations
- Identify churn patterns

Key Insights:
- Customers with month-to-month contracts have higher churn
- Higher monthly charges increase churn probability
- Long-term contracts reduce churn risk

Visualizations were created using Matplotlib and Seaborn.

---

## ⚙️ Data Preprocessing & Feature Engineering

Steps performed:
- Handling missing values
- Encoding categorical features
- Scaling numerical features
- Train-test split
- Feature transformation for model compatibility

---

## 🤖 Machine Learning Models Used

The following models were trained and evaluated:

- Logistic Regression (Baseline model)
- Decision Tree Classifier
- Random Forest Classifier
- AdaBoost Classifier
- XGBoost Classifier
- Optuna Tuned Model (Hyperparameter optimization)

---

## 📈 Model Evaluation

Models were evaluated using:
- Accuracy Score
- Confusion Matrix
- Precision, Recall, and F1-score

Ensemble and boosted models performed better in capturing complex churn patterns.

---

## 🖥 Streamlit Application

A Streamlit web application is included that:
- Accepts customer input details
- Loads the trained model
- Predicts churn in real time
- Displays results in a user-friendly format

---

## 🛠 Tech Stack

Programming Language:
- Python

Libraries & Tools:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- optuna
- streamlit

Development Tools:
- Jupyter Notebook
- VS Code

---

## 📁 Project File Structure

```
Telecom_Customer_Churn_Prediction_using_Machine_Learning/
├── Churn_Analysis_EDA.ipynb
│ └── Exploratory data analysis and visualization
├── ML_Model_Building.ipynb
│ └── Model training, evaluation, and model selection
├── Customer-Churn.csv
│ └── Telecom customer dataset
├── streamlit_app.py
│ └── Streamlit web application for churn prediction
├── best_xgboost_churn_model.pkl
│ └── Trained XGBoost model
├── best_optuna_churn_model.pkl
│ └── Hyperparameter optimized model (Optuna)
├── ada_boost_churn_model.pkl
│ └── AdaBoost trained model
├── requirements.txt
│ └── Project dependencies
└── README.md
└── Project documentation
```

---

## 🚀 How to Run the Project

1. Clone the repository  
   git clone https://github.com/Musharraf-Bubere/Telecom_Customer_Churn_Prediction_using_Machine_Learning.git

2. Navigate to the project folder  
   cd Telecom_Customer_Churn_Prediction_using_Machine_Learning

3. Install dependencies  
   pip install -r requirements.txt

4. Run the Streamlit app  
   streamlit run streamlit_app.py

---

## 🔮 Future Enhancements

- Handle class imbalance using SMOTE
- Add Power BI or Plotly dashboards
- Deploy using Flask or FastAPI
- Integrate database storage
- Improve UI/UX of Streamlit app

---

## 👤 Author

Musharraf Bubere  
Aspiring Data Analyst | Machine Learning Enthusiast  

GitHub: https://github.com/Musharraf-Bubere  
LinkedIn: https://www.linkedin.com/in/musharraf-bubere007/

---

⭐ If you found this project useful, don’t forget to give it a star on GitHub!

