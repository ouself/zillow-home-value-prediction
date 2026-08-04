# Zillow Home Value Prediction

Predicting residential property tax valuations using structured property data, comparing three regression approaches through a full feature engineering, selection, and hyperparameter tuning pipeline.

## Dataset 

Zillow property dataset (2017 Kaggle Zestimate competition data), accessed via course repository at https://www.cs.bu.edu/fac/snyder/cs505/Data/zillow_dataset.csv. Download and place in the project root as zillow_dataset.csv before running the notebook.

## Problem

Given ~78,000 property records with features like square footage, location, and structural attributes, predict the assessed property tax value (`taxvaluedollarcnt`). The dataset spans a wide range ($1,000 to nearly $49 million) with heavy right-skew, making this a genuinely challenging regression problem.

## Approach

- **Data cleaning:** Removed unusable columns (unique IDs, constant-value features, columns missing >70% of data), imputed remaining gaps using type-appropriate strategies (mode for categorical IDs, zero for garage features, etc.)
- **Feature engineering:** Tested log-transformation, redundant-feature removal, and scaling individually across all three models using repeated 5-fold cross-validation (25 total folds), rather than assuming any preprocessing step helps universally
- **Models compared:** Ridge Regression, Decision Tree, and HistGradientBoostingRegressor, selected to represent distinct modeling approaches (regularized linear, single tree, gradient-boosted ensemble)
- **Feature selection:** Forward selection (Ridge) and permutation importance (HistGradientBoosting), applied selectively where each method actually improved performance
- **Hyperparameter tuning:** Grid search with repeated cross-validation for all three models

## Key Findings

- **Final model:** HistGradientBoostingRegressor, achieving a held-out test MAE of **$209,570**
- Log-transforming the skewed target improved Ridge Regression's CV MAE by **~9%**, but slightly *hurt* HistGradientBoosting — a clear demonstration that preprocessing benefits are model-specific, not universal
- Decision Tree overfit severely by default (training MAE ~$2,100 vs. CV MAE ~$262,000); constraining `max_depth` and `min_samples_leaf` resolved this, improving CV MAE by over $47,000
- Identified and corrected a data-consistency bug during hyperparameter tuning, where a model's hyperparameter search and final evaluation were inadvertently run on two different feature sets

## Tech Stack

Python, pandas, scikit-learn (Ridge, DecisionTreeRegressor, HistGradientBoostingRegressor, GridSearchCV, cross_val_score), matplotlib/seaborn

## Note

This project was completed as part of a 3-person team for a graduate Machine Learning Fundamentals course. This repository reflects my individual contributions to data cleaning, feature engineering, and baseline/final model evaluation, developed collaboratively with two teammates on feature selection and hyperparameter tuning.
