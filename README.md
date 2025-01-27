# Experimenting with Fine-Tuning ML Models: XGBoost, Gradient Boosting, and More

This project was a personal exploration into fine-tuning machine learning models to see how **XGBoost** compares to **Gradient Boosting** and other popular algorithms. My goal was to understand the nuances of these models and evaluate their performance on a churn prediction dataset.

## Goals
- Fine-tune **XGBoost** and compare it with other boosting algorithms like **Gradient Boosting**, **AdaBoost**, and **CatBoost**.
- Experiment with both boosting and non-boosting models to see how they stack up.
- Optimize models to achieve the best balance between **accuracy**, **precision**, and **recall**.

## Steps I Took

### 1. **Data Preprocessing**
- Handled missing values by dropping incomplete rows.
- One-hot encoded categorical variables like `Geography` and `Gender`.
- Used **SMOTE** to deal with class imbalance in the dataset.
- Scaled features using `StandardScaler`.

### 2. **Baseline Models**
Trained several baseline models to establish a performance benchmark:
- **Decision Tree**
- **K-Nearest Neighbors**
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine**
- **XGBoost**
- **Gradient Boosting**
- **AdaBoost**
- **LightGBM**
- **CatBoost**

### 3. **Fine-Tuning Focus**
- Performed hyperparameter tuning for **XGBoost** and **Random Forest** using `GridSearchCV`.
- Tested different combinations of learning rates, max depth, and the number of estimators for **XGBoost**.
- Compared the tuned **XGBoost** to **Gradient Boosting** and other models to assess which performed best.

### 4. **Dimensionality Reduction**
- Applied **PCA** to reduce the feature space while preserving 95% variance.
- Evaluated how dimensionality reduction affected models like **Logistic Regression**.

### 5. **Performance Metrics**
Evaluated models based on:
- **Accuracy**
- **Precision**
- **Recall**

## Results
- **Gradient Boosting** emerged as the top-performing model with the highest accuracy and precision.
- Fine-tuned **XGBoost** was a close second, showing robust performance after optimization.
- **Random Forest** (tuned) also performed exceptionally well, making it a strong, interpretable alternative.

| Model                        | Accuracy  | Precision | Recall  |
|------------------------------|-----------|-----------|---------|
| Gradient Boosting            | 0.998535  | 0.999581  | 0.997488 |
| XGBoost (Tuned)              | 0.997698  | 0.998324  | 0.997070 |
| Random Forest (Tuned)        | 0.998326  | 0.999161  | 0.997488 |
| Logistic Regression (PCA)    | 0.992256  | 0.997462  | 0.987024 |

## Conclusion
- **Gradient Boosting** is the most balanced and reliable model for this dataset, achieving the highest overall performance.
- **XGBoost** and **Random Forest** are strong alternatives, with **XGBoost** excelling after fine-tuning.
- Models like **Logistic Regression with PCA** showed slightly lower performance but are still effective in scenarios where simplicity is preferred.

This project helped me better understand the strengths and trade-offs of boosting models like XGBoost, Gradient Boosting, and CatBoost in comparison to other algorithms.
