# NeuroFive ML Track — Week 4

## Task 1: Build a Proper ML Pipeline with Feature Engineering

### Overview

This task moved away from manually chaining preprocessing steps and toward the industry-standard approach: scikit-learn Pipelines. A pipeline bundles preprocessing and modeling into a single object, ensuring transformations are applied consistently and preventing data leakage between training and test sets.

### Approach

I returned to the Titanic dataset and built a `ColumnTransformer` that applies `StandardScaler` to numerical columns (`Age`, `Fare`, `SibSp`, `Parch`, `FamilySize`) and `OneHotEncoder` to categorical columns (`Pclass`, `Sex`, `Embarked`, `IsAlone`). This transformer was chained together with a Logistic Regression classifier into a single `Pipeline` object, so the entire workflow — from raw features to prediction — fits and evaluates as one unit.

I also engineered two new features to test whether they improved the model: `FamilySize`, calculated as the sum of siblings/spouses and parents/children plus the passenger themselves, and `IsAlone`, a binary flag indicating whether a passenger was traveling solo. To test their impact fairly, I built two versions of the pipeline — one with these engineered features included, and one without — keeping every other step identical.

Finally, I saved the completed pipeline using `joblib`, allowing it to be reloaded and reused for predictions without needing to retrain from scratch.

### Results

| Version | Accuracy |
|---|---|
| Without FamilySize / IsAlone | 79.89% |
| With FamilySize / IsAlone | 79.33% |

Adding `FamilySize` and `IsAlone` produced a slight *decrease* in accuracy of about half a percentage point, rather than an improvement. This is a realistic and useful outcome: not every engineered feature adds value, and in this case the new features likely overlapped with information already captured by `SibSp` and `Parch`, which remained in the model. Rather than discard this finding, I've kept it as evidence that feature engineering should always be validated against a baseline rather than assumed to help.

---

## Task 2: Ensemble Learning — Random Forest vs XGBoost

### Overview

This task explored ensemble methods, which combine multiple models to produce stronger and more reliable predictions than any single model alone. I trained and compared a Random Forest and an XGBoost classifier against my earlier Logistic Regression baseline, all on the same Titanic dataset.

### Approach

Using the same cleaned and encoded dataset and train/test split as previous weeks, I trained a `RandomForestClassifier` and an `XGBClassifier`, evaluating both against the original Logistic Regression model on accuracy, precision, recall, and F1-score. I then compared feature importances between the two ensemble models to see whether they prioritized different features when making predictions.

### How Random Forest and XGBoost Differ

Random Forest builds many decision trees independently and in parallel, each trained on a random subset of the data and features, then averages their predictions to reduce overfitting — an approach known as bagging. XGBoost instead builds trees sequentially, where each new tree is trained specifically to correct the errors made by the trees before it — an approach known as boosting. Because of this, XGBoost often achieves higher accuracy on structured, tabular data like this dataset, but it is more sensitive to hyperparameter choices and can overfit more easily if not tuned carefully. Random Forest tends to be more robust out-of-the-box with less tuning required, making it a reliable baseline before reaching for a more complex boosted model.

### Results

*[Comparison table (Logistic Regression vs Random Forest vs XGBoost — Accuracy, Precision, Recall, F1-score) and top 3 feature importances for each ensemble model to be added once final values are confirmed.]*

### Tools and Libraries

- Python 3.10.5, Visual Studio Code with the Jupyter extension
- pandas, numpy, scikit-learn, xgboost, joblib, matplotlib

### Repository Contents

- `Week4_ML_Pipeline.ipynb` — pipeline construction, feature engineering, and evaluation
- `Week4_Ensemble_Learning.ipynb` — Random Forest vs XGBoost comparison
- `titanic_pipeline.pkl` — the final saved pipeline
- `README.md` — this document

### Next Steps

With ensemble methods now benchmarked against simpler models, future work could combine both directions from this week — running the winning ensemble model through the same pipeline structure built in Task 1, and applying it to the churn or housing datasets from earlier weeks.

---
*Part of the NeuroFive Solutions ML Track — Week 4: Machine Learning Fundamentals*
