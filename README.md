# 🚢 Titanic Survival Prediction — End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.x-green) ![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-blue)

This project demonstrates the development of an end-to-end Machine Learning pipeline for predicting passenger survival on the Titanic dataset using Scikit-learn. The workflow integrates data preprocessing, missing value imputation, categorical encoding, feature scaling, feature selection, model training, cross-validation, and model serialization into a single reusable Pipeline object.

The trained pipeline can be saved and deployed for inference on new passenger data without requiring separate preprocessing steps.

## Skills Demonstrated

![Python](https://img.shields.io/badge/Python-3.x-green)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-blue)
![Pipelines](https://img.shields.io/badge/Pipelines-End--to--End-orange)
![Feature Engineering](https://img.shields.io/badge/Feature-Engineering-yellow)
![Cross Validation](https://img.shields.io/badge/Cross_Validation-5_Fold-success)
![Decision Tree](https://img.shields.io/badge/Decision_Tree-Classifier-informational)
![Pickle](https://img.shields.io/badge/Model-Persistence-lightgrey)


## Pipeline architecture

```
Imputation → OneHotEncoding → MinMaxScaler → SelectKBest(chi2, k=5) → DecisionTreeClassifier
```

| Step | Transformer | What it does |
|------|-------------|--------------|
| trf1 | ColumnTransformer | Mean impute Age, most-frequent impute Embarked |
| trf2 | OneHotEncoder | Encode Sex and Embarked |
| trf3 | MinMaxScaler | Scale all features to [0, 1] |
| trf4 | SelectKBest(chi2) | Keep top 5 features |
| trf5 | DecisionTreeClassifier | Final estimator |

## Results

| Metric | Score |
|--------|-------|
| Test accuracy | 62.6% |
| Cross-validated accuracy (5-fold) | 63.9% |
| Features selected | 5 of 10 |

## Repository structure

```
├── part_1_pipeline.ipynb   # Preprocessing + training
├── part_2_predict.ipynb    # Inference with saved model
├── titanic_pipe.pkl        # Serialised pipeline
└── train.csv               # Titanic dataset
```

## Quick start

```bash
git clone https://github.com/MR-ROGUE01/end-to-end-titanic-survival-pipeline
cd end-to-end-titanic-survival-pipeline
pip install scikit-learn pandas numpy
```

## Inference

Load the serialised pipeline and pass a DataFrame with the 7 input features:

```python
import pickle, pandas as pd

pipe = pickle.load(open('titanic_pipe.pkl', 'rb'))

passenger = pd.DataFrame({
    'Pclass': [1], 'Sex': ['female'], 'Age': [29.0],
    'SibSp': [0], 'Parch': [0], 'Fare': [211.3], 'Embarked': ['S']
})

pipe.predict(passenger)  # → array([1])
```

## Features used

Input features (after dropping PassengerId, Name, Ticket, Cabin):
`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`

Top 5 selected by chi² after encoding and scaling.

## Possible improvements

- Swap DecisionTree for RandomForest or GradientBoosting for better generalisation
- Add GridSearchCV wrapping the full pipeline for hyperparameter tuning
- Engineer new features: title from Name, family size from SibSp + Parch
- Use IterativeImputer for smarter age imputation

---

Built as a learning project. Contributions and suggestions welcome via issues or PRs.
