# Stroke Prediction – Final Machine Learning Project

**Author:** Víctor Asem 
**Date:** December 2025  
**Course:** Machine Learning

## Objective
Predict the probability of suffering a stroke using demographic and clinical features.

## Dataset
- Source: [Kaggle – Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- 5,110 records – highly imbalanced (only **4.87 %** positive stroke cases)

## Winner Model
**AdaBoost**  
- **F1-score:** **0.2877** (best of all models)  
- Recall: **42 %**  
- Precision: **21.88 %**  
- AUC-ROC: **0.8279**  
- Optimal threshold: **0.45**

AdaBoost outperforms both Logistic Regression (F1 = 0.2305) and the tuned Random Forest (F1 = 0.2647).

## Notebook Structure
1. Data loading & provenance  
2. Exploratory Data Analysis (EDA) + visualizations  
3. Data cleaning (gender='Other' removal, BMI median imputation, id drop)  
4. Preprocessing pipeline (StandardScaler + OneHotEncoder)  
5. Model comparison: Logistic Regression, Random Forest (GridSearchCV + threshold tuning), AdaBoost  
6. Final results & winning model selection

## Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
