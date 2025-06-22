# Lung_Cancer_ML

# About the Project:

- This research uses **machine learning methods** to improve the **early detection of lung cancer**.

- The study uses a **detailed synthetic medical dataset** containing records from **22,811 patients**.

- Each patient record includes **788 health-related parameters**, covering both **numerical and categorical** data.

- The **target variable** indicates whether a patient has **lung cancer** (presence or absence).

- The dataset underwent **extensive filtering**, and **missing values** were filled using the **Random Forest imputation** method.

- After preprocessing, the dataset was reduced to **89 key features**.

- **Feature selection and dimensionality reduction** were carried out using:
  - **Principal Component Analysis (PCA)**
  - **Brain Storm Optimization (BSO)**
  - **Recursive Feature Elimination (RFE)**
  - **SelectKBest (SelectK)**

- Four machine learning models were trained on each of the transformed datasets:
  - **XGBoost**
  - **Support Vector Machine (SVM)**
  - **CatBoost**
  - **K-Nearest Neighbors (KNN)**

- Model performance was evaluated using **5-fold cross-validation**, focusing on key metrics like **accuracy** and **recall**.

- An **ensemble model** was developed to combine the outputs of the individual models to improve **overall predictive performance and reliability**.

- Among all feature selection methods, **RFE** yielded the best results in the ensemble model:
  - **Accuracy**: 98.746%
  - **Recall**: 96.245%
  - **Precision**: 98.582%
  - **F1-score**: 97.4%

- The research concludes that applying machine learning to predict early-stage lung cancer can **significantly improve survival rates** and help **reduce medical costs**.

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
