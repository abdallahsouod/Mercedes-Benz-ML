# 🚗 Mercedes-Benz Testing Time Prediction

## 📌 Project Overview

This project is an end-to-end Machine Learning solution for predicting the testing time of Mercedes-Benz vehicles during the manufacturing process.

The main objective is to predict the time required to test a vehicle, measured in seconds, based on its anonymized configuration features.

The project focuses strongly on **Feature Shrinkage**, comparing Lasso-based feature selection with PCA dimensionality reduction, while also evaluating several regression models to identify the best-performing approach.

---

## 🎯 Business Problem

Different vehicle configurations may require different testing durations during the manufacturing process.

Accurately predicting testing time can help:

* Identify potential production bottlenecks.
* Improve manufacturing planning.
* Estimate testing workload.
* Support production-line optimization.
* Reduce uncertainty around testing duration.

---

## 📊 Dataset

The dataset contains anonymized vehicle configuration features.

### Training Data

* Rows: **4,209**
* Columns: **378**
* Target variable: `y`
* Identifier: `ID`

### Test Data

* Rows: **4,209**
* Columns: **377**

The target variable `y` represents the vehicle testing time in seconds.

### Feature Types

The dataset contains:

* **8 categorical features**

  * `X0`
  * `X1`
  * `X2`
  * `X3`
  * `X4`
  * `X5`
  * `X6`
  * `X8`

* The remaining features are mainly binary numerical configuration features.

The original feature space contained **376 features excluding ID and target**.

---

# 🔍 Exploratory Data Analysis

Several data quality and exploratory analysis steps were performed before modeling.

## Data Quality Checks

### Missing Values

No missing values were found in either the training or test dataset.

```text
Train Missing Values: 0
Test Missing Values: 0
```

### Duplicate Rows

```text
Train Duplicate Rows: 0
Test Duplicate Rows: 0
```

### Duplicate IDs

```text
Train Duplicate IDs: 0
Test Duplicate IDs: 0
```

### Constant Features

A total of **12 constant features** were identified.

These features contained no useful variation and were removed during cleaning.

### Duplicate Features

A total of **56 redundant/identical features** were identified during the data audit.

These redundant features were removed as part of the cleaning stage.

---

# 🧹 Data Cleaning

After removing redundant and constant features:

```text
Original Features
376
    ↓
After Cleaning
319
```

The target variable outlier analysis was also performed.

Using the IQR method:

* Q1 = **90.82**
* Q3 = **109.01**
* IQR = **18.19**
* Upper Bound = **136.295**
* Statistical Outliers = **50**

The identified outliers were **retained** because they may represent legitimate vehicle configurations rather than data errors.

---

# ⚙️ Preprocessing

The preprocessing pipeline was designed to handle both categorical and numerical features.

### Categorical Features

Categorical variables were transformed using:

```text
OneHotEncoder
```

with:

```text
handle_unknown="ignore"
```

### Numerical Features

Numerical variables were scaled using:

```text
StandardScaler
```

### Feature Space

The preprocessing stage expanded the feature space:

```text
319 Clean Features
        ↓
One-Hot Encoding + Scaling
        ↓
504 Processed Features
```

---

# 🧠 Machine Learning Models

Several regression algorithms were evaluated.

## Linear Regression

* MAE: **5.727**
* RMSE: **8.359**
* R²: **0.551**

## Lasso Regression

Best validation result:

* Alpha ≈ **0.3857**
* MAE: **5.388**
* RMSE: **8.028**
* R²: **0.586**
* Non-zero features: **22**

Lasso reduced the processed feature space from:

```text
504 → 22 Features
```

This represents approximately:

```text
95.6% Feature Reduction
```

This result demonstrates the effectiveness of **Feature Shrinkage**.

## Ridge Regression

* MAE: **5.694**
* RMSE: **8.315**
* R²: **0.556**

## Random Forest

* MAE: **5.919**
* RMSE: **9.160**
* R²: **0.461**

## Gradient Boosting

Final best-performing model:

* MAE: **5.246**
* RMSE: **7.873**
* R²: **0.602**

Final parameters:

```text
n_estimators = 100
learning_rate = 0.05
max_depth = 2
random_state = 42
```

## PCA + Ridge

As a bonus experiment, PCA was used to reduce dimensionality while retaining 95% explained variance.

Results:

* MAE: **5.570**
* RMSE: **8.255**
* R²: **0.562**

PCA successfully reduced dimensionality but performed worse than Lasso in this experiment.

---

# 🏆 Final Model

**Gradient Boosting Regressor** was selected as the final prediction model because it achieved the best validation performance.

### Final Performance

| Metric |    Result |
| ------ | --------: |
| MAE    | **5.246** |
| RMSE   | **7.873** |
| R²     | **0.602** |

The final model was retrained using the complete training dataset and used to generate predictions for all **4,209 test samples**.

---

# 🔥 Feature Shrinkage

One of the main objectives of this project was to investigate Feature Shrinkage.

The results were:

```text
376 Original Features
        ↓
319 After Cleaning
        ↓
504 After Encoding
        ↓
22 Lasso Non-Zero Features
```

Lasso therefore removed approximately:

**95.6% of the processed feature space.**

Despite the significant reduction in dimensionality, Lasso maintained strong predictive performance with:

```text
RMSE = 8.028
R² = 0.586
```

---

# 📐 PCA vs Lasso

The project also compared PCA with Lasso.

| Method      | Dimensions / Selected Features |      RMSE |        R² |
| ----------- | -----------------------------: | --------: | --------: |
| Lasso       |       **22 selected features** | **8.028** | **0.586** |
| PCA + Ridge |             Reduced components |     8.255 |     0.562 |

### Conclusion

Lasso provided a better balance between:

* Prediction performance
* Feature reduction
* Interpretability

PCA was useful for dimensionality reduction, but Lasso achieved better validation performance in this experiment.

---

# 🌐 Streamlit Application

A Streamlit application was developed to provide an interactive interface for the final Machine Learning pipeline.

The application contains four main pages:

## 🏠 Home

Provides:

* Project overview
* Business problem
* Dataset statistics
* Key model results
* Feature Shrinkage summary

## ⚡ Live Prediction

Allows users to:

* Select a vehicle configuration from the test dataset.
* Run the trained model.
* View the predicted testing time in seconds.

## 📊 Model Comparison

Displays:

* MAE comparison
* RMSE comparison
* R² comparison
* Feature Shrinkage results
* Model benchmarking

## 📈 Analytics Dashboard

Provides interactive visualizations including:

* Prediction distribution
* Model performance comparison
* Feature Shrinkage visualization
* Categorical feature complexity
* Target distribution
* Model insights
* Prediction CSV download

---

# 💾 Saved Model Artifacts

The final deployment artifacts include:

```text
models/
├── final_gradient_boosting.joblib
└── preprocessor.joblib
```

The generated test predictions are stored in:

```text
test_predictions.csv
```

---

# 📁 Project Structure

```text
Mercedes-Benz-ML/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── models/
│   ├── final_gradient_boosting.joblib
│   └── preprocessor.joblib
│
├── notebook/
│   └── Mercedes_Benz_ML.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── test_predictions.csv
```

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Plotly
* Streamlit

---

# 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Mercedes-Benz-ML
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

---

# 📌 Project Highlights

* End-to-end Machine Learning pipeline.
* Complete data quality audit.
* No missing values or duplicate rows.
* Feature redundancy reduction.
* One-Hot Encoding and Scaling pipeline.
* Lasso Feature Shrinkage.
* 95.6% reduction in processed features.
* Model benchmarking across multiple regression algorithms.
* PCA vs Lasso comparison.
* Gradient Boosting achieved the best validation RMSE.
* Interactive Streamlit deployment dashboard.
* Batch predictions generated for 4,209 test configurations.

---

# 👨‍💻 Project Goal

The project demonstrates how Machine Learning can be used to predict manufacturing testing time while balancing:

**Prediction Accuracy + Feature Reduction + Interpretability + Deployment**

The final solution combines robust preprocessing, model comparison, Feature Shrinkage, and an interactive Streamlit application to deliver an end-to-end manufacturing analytics solution.
