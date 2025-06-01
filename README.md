# Comparing-Classifiers  

**Practical Application Assignment 17.1**

---

## Contents
- [Introduction](#introduction)  
- [How to Use This Repository](#how-to-use-this-repository)  
- [Business Understanding](#business-understanding)  
- [Data Understanding](#data-understanding)  
- [Data Preparation](#data-preparation)  
- [Baseline Model Comparison](#baseline-model-comparison)  
- [Model Comparisons](#model-comparisons)  
- [Improving the Model](#improving-the-model)  
- [Findings](#findings)  
- [Next Steps and Recommendations](#next-steps-and-recommendations)   

---

## Introduction

### Overview
In this third practical application assignment, the goal is to compare the performance of several classification algorithms—**K-Nearest Neighbors**, **Logistic Regression**, **Decision Trees**, and **Support Vector Machines**—introduced in this section of the program. These models are applied to a real-world dataset involving the marketing of long-term deposit products over the phone.

### Data
The dataset is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing) and comes from a Portuguese banking institution. It includes customer information and the outcomes of multiple direct marketing campaigns. The target variable indicates whether a client subscribed to a term deposit. The goal is to use this information to train models that predict the likelihood of subscription.

This project includes data cleaning, feature transformation, hyperparameter tuning using grid search, and model evaluation based on accuracy to compare performance and derive actionable insights.

### Classifiers Compared
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Decision Tree  
- Support Vector Machine (SVM)

---

## How to Use This Repository

1. Clone the repository.  
2. Open `Classifier_Comparison.ipynb` using Jupyter Notebook or Jupyter Lab.  
3. Ensure the following libraries are installed:
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `matplotlib`
   - `seaborn`

---

## Business Understanding

The dataset reflects the results of direct marketing campaigns run by a Portuguese bank. Each campaign involved phone calls to clients, aiming to convince them to subscribe to a long-term deposit product.

### Objectives
- Use classification algorithms to predict whether a customer will subscribe.
- Identify patterns among features that correlate with positive outcomes.
- Provide insight to guide future marketing efforts and improve targeting.

---

## Data Understanding

The dataset is well-structured and contains no missing values. It includes categorical and numerical features such as job type, marital status, education level, loan status, and communication method.

### Key Observations
- The dataset is imbalanced, with a majority of "no" responses.
- Customers with housing loans showed a higher success rate (52.4%).
- No explicit gender variable is included; marital status is the closest proxy but is not equivalent.
- Customers contacted via cellular phone were more likely to subscribe.

---

## Data Preparation

The following steps were taken to prepare the data:
- Renamed the target column from `y` to `deposit`.
- Selected relevant features: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`.
- Applied one-hot encoding and standard scaling using `ColumnTransformer`.
- Encoded the target using `LabelEncoder`.
- Split the data into training and testing sets using a 70/30 split.

---

## Baseline Model Comparison

Initial models were built using default parameters. The results below compare Decision Tree and Logistic Regression:

| Model               | Accuracy | Precision | Recall | F1 Score | Fit Time (ms) |
|---------------------|----------|-----------|--------|----------|----------------|
| Decision Tree       | 0.8875   | 0.4438    | 0.5000 | 0.4702   | 128            |
| Logistic Regression | 0.8876   | 0.4438    | 0.5000 | 0.4702   | 193            |

Both models performed similarly in terms of accuracy. However, due to class imbalance, precision, recall, and F1 scores were moderate.

---

## Model Comparisons

Performance of models using default hyperparameters:

| Model               | Train Time (s) | Train Accuracy | Test Accuracy |
|---------------------|----------------|----------------|----------------|
| Logistic Regression | 0.322          | 0.8872         | 0.8876         |
| KNN                 | 55.8           | 0.8846         | 0.8808         |
| Decision Tree       | 0.376          | 0.8912         | 0.8848         |
| SVM                 | 24.4           | 0.8873         | 0.8875         |

Logistic Regression and Decision Tree models were the most efficient and accurate, with fast training times and strong test performance.

---

## Improving the Model

To improve model performance, hyperparameter tuning was applied using `GridSearchCV` with 10-fold cross-validation:

| Model               | Train Time (s) | Best Parameters                                               | Best CV Score |
|---------------------|----------------|----------------------------------------------------------------|----------------|
| Logistic Regression | 64             | `C=0.001`, `penalty='l2'`, `solver='liblinear'`                | 0.8872         |
| KNN                 | 302            | `n_neighbors=17`                                               | 0.8855         |
| Decision Tree       | 15.7           | `criterion='entropy'`, `max_depth=1`, `min_samples_leaf=1`     | 0.8872         |
| SVM                 | 490            | `C=0.1`, `kernel='rbf'`                                        | 0.8872         |

All models (except KNN) achieved similar cross-validation scores. The Decision Tree was the fastest to train and matched the top-performing models in accuracy.

---

## Findings

- Logistic Regression and Decision Tree classifiers consistently performed well with minimal training time.
- The dataset is highly imbalanced, which affected precision and recall across all models.
- No gender feature was available; marital status was not used as a substitute.
- Housing loan status was the strongest individual predictor of a positive response.
- Contact method (cellular phone) correlated with higher subscription success.

---

## Next Steps and Recommendations

- Introduce performance metrics beyond accuracy (e.g., F1 score, ROC AUC) for a better view of model behavior.
- Address class imbalance using resampling (e.g., SMOTE) or class weighting.
- Explore feature selection and engineering to improve model generalizability.
- Refine targeting strategies based on key predictors such as loan status and contact method.

---
## Link to Jupyter Notebook

[Click here to view the notebook](https://github.com/walhathal/Comparing-Classifiers/blob/main/prompt_III.ipynb)

## Author

Wael Alhathal - University of California, Berkeley - AI/ML Student
