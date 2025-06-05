# Data_analysis_and_visualization_laptop_pricing
# 💻 Laptop Pricing Analysis & Prediction

This project focuses on data analysis, visualization, and regression modeling using a dataset of laptop features and their prices. The goal is to explore patterns and predict laptop prices based on various attributes using different regression techniques.

---

## 📦 Dataset

The dataset is sourced from IBM's Developer Skills Network on Coursera and is available at:
https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod1.csv


---

## 📊 Key Features of the Project

### 1. **Data Cleaning & Preprocessing**
- Handling missing values using mean imputation.
- Removing or converting incorrect datatypes.
- Standardizing and normalizing feature scales.

### 2. **Data Transformation**
- Converting screen size and weight to standard units (inches and pounds).
- Normalizing CPU frequency.
- Binning prices into categories (low, medium, high).
- Creating dummy variables for categorical data.

### 3. **Exploratory Data Analysis (EDA)**
- Regression plots for numerical features vs price.
- Box plots for categorical features vs price.
- Heatmaps via pivot tables.

### 4. **Regression Modeling**
- **Simple Linear Regression**
- **Multiple Linear Regression**
- **Polynomial Regression**
- **Pipeline Implementation**
- **Ridge Regression** (with Grid Search and Cross-Validation)

### 5. **Model Evaluation**
- R-squared score and Mean Squared Error (MSE)
- Cross-validation scores
- Overfitting checks using polynomial degree variation

---

## 🛠 Libraries Used

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `sklearn` (LinearRegression, Ridge, PolynomialFeatures, StandardScaler, Pipeline, train_test_split, GridSearchCV, etc.)

---

## 📈 Sample Outputs

- Plots showing regression fits, KDE comparisons, and price bin distributions.
- R-squared values comparing different polynomial degrees.
- Ridge Regression with Grid Search to tune hyperparameters.

---

## 📁 How to Run

1. Download the dataset (already accessed via URL in the script).
2. Install required Python packages if not already installed:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
