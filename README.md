# Lung_Cancer_ML

# About the Project:

This research employs machine learning methods to enhance early lung cancer detection
using a detailed synthetic medical dataset. The dataset contains records from 22,811 patients,
each with 788 health-related parameters, comprising both numeric and categorical data. The
target variable denotes lung cancer presence or absence. Extensive data filtering was done,
and imputation was done using the **Random Forest** approach. The dataset was condensed to
89 key features. Feature selection and reduction were performed using **Principal Component
Analysis (PCA)**, **Brain Storm Optimization (BSO)**, **Recursive Feature Elimination (RFE)**, and
**SelectKBest (SelectK)**. Four machine learning models—**XGBoost**, **Support Vector Machine
(SVM)**, **CatBoost**, and **K-Nearest Neighbors (KNN)**—were trained on each of these
transformed datasets. The model's performance was assessed through **5-fold cross-validation**,
focusing on accuracy and recall. An **ensemble model** was then constructed to combine
individual model outputs, aiming to improve overall predictive accuracy and reliability. This
study seeks to identify the most effective feature selection method, thereby enhancing early
detection capabilities for lung cancer. The results showed that **RFE** was the most effective
feature selection algorithm, resulting in the highest accuracy (98.746%), recall (96.245%),
precision (98.582%), and F1-score (97.4%) within the ensemble model. The project
concluded that using machine learning models to predict the early risk of lung cancer has the
potential to significantly improve survival chances and reduce medical costs.

# Flowchart of the Methodology Used:

![image](https://github.com/user-attachments/assets/05af1faf-8309-4b71-a39c-4703f2deba34)


# Results:

![image](https://github.com/user-attachments/assets/d1096f4d-77f8-4d3c-8eb9-51293f919ca5)

![image](https://github.com/user-attachments/assets/b5609304-887a-4f83-89e1-e84a9475f87e)

# Future Work:

- Exploring deep learning models such as LSTM and Tab-R for training our early
detection model for lung cancer.
- Validating the synthetic dataset features with real-world Electronic Medical Records
(EMR) data.
- Translating the model into a user-friendly and accessible website for public use,
aimed at detecting early stages of lung cancer, ensuring it is both error-free and
visually appealing.
