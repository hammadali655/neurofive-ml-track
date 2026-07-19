# NeuroFive ML Track — Week 5

## Task 1: Handling Imbalanced & Messy Real-World Data

### Overview

Real-world datasets are rarely balanced, and letting that imbalance go unaddressed can quietly produce a model that looks accurate but fails at the task that actually matters. This task revisited the Telco Customer Churn dataset from Week 3 to properly diagnose and handle class imbalance, rather than training on it blindly.

### Approach

After cleaning the dataset (converting `TotalCharges` to numeric and encoding categorical features), I checked the class balance of the target variable and visualized it with a bar chart, confirming that roughly 73% of customers did not churn versus 27% who did — a meaningful but not extreme imbalance.

I trained a baseline Logistic Regression model with no imbalance handling, then applied two different techniques to address it: SMOTE (Synthetic Minority Oversampling Technique, via `imbalanced-learn`), which generates synthetic examples of the minority class within the training set only, and `class_weight='balanced'`, which reweights the loss function to penalize misclassifying the minority class more heavily. Both approaches were compared against the baseline using Precision, Recall, and F1-score specifically for the churn class.

### Results

| Approach | Accuracy | Precision (Churn) | Recall (Churn) | F1-score (Churn) |
|---|---|---|---|---|
| Before (Baseline) | 80.31% | 0.646 | 0.575 | 0.608 |
| After SMOTE | 75.98% | 0.541 | 0.639 | 0.586 |
| After class_weight='balanced' | 72.71% | 0.492 | 0.794 | 0.607 |

Both rebalancing techniques traded overall accuracy for a meaningful gain in recall — the metric that matters most for this problem, since a missed churner represents a customer the business never gets the chance to try to retain. `class_weight='balanced'` pushed recall the furthest, from 57.5% to 79.4%, meaning the model catches substantially more true churners, at the cost of more false alarms (lower precision) and a drop in overall accuracy. SMOTE landed in between, offering a smaller recall gain with a smaller accuracy trade-off. Which approach is "better" depends on the business cost of a missed churner versus a false alarm — for retention campaigns, where reaching out to a loyal customer by mistake is far cheaper than losing a real one, the higher-recall `class_weight='balanced'` model is likely the more useful choice in practice.

### Why Accuracy Alone Is Misleading Here

With roughly 73% of customers not churning, a model could predict "No Churn" for every customer and still score around 73% accuracy while being completely useless for identifying at-risk customers. Accuracy treats every correct prediction equally, but in this problem the cost of missing a real churner is much higher than the cost of a false alarm, since a missed churner represents lost revenue the business never gets a chance to try to prevent. This is exactly what the results above demonstrate: the baseline model had the highest accuracy of the three, but also the weakest recall — meaning it looked the "best" on paper while actually catching the fewest real churners.

---

## Task 2: Deploy Your Model as a Live Web App

### Overview

A model in a notebook has no real-world impact until someone can actually use it. This task took the Titanic survival pipeline built in Week 4 and turned it into a live, interactive web application using Streamlit.

### Approach

I loaded the saved `titanic_pipeline.pkl` — which bundles all preprocessing (scaling and encoding) together with the trained Logistic Regression model — inside a Streamlit app. The app presents input fields for passenger class, sex, age, fare, siblings/spouses aboard, parents/children aboard, and port of embarkation, automatically calculates the engineered `FamilySize` and `IsAlone` features to match the training pipeline, and returns a survival prediction along with a probability score when the user clicks "Predict Survival." The interface was styled with custom CSS for a clean, single-screen layout, and the app was deployed publicly using Streamlit Community Cloud.

### Live App

*[Live Streamlit Cloud link to be added here once deployment is confirmed.]*

### Tools and Libraries

- Python 3.10.5, Visual Studio Code with the Jupyter extension
- pandas, numpy, scikit-learn, imbalanced-learn, streamlit, joblib

### Repository Contents

- `Week5_Imbalanced_Data.ipynb` — churn class imbalance analysis and SMOTE/class-weight comparison
- `streamlit_app/app.py` — the deployed Titanic survival prediction web app
- `streamlit_app/requirements.txt` — deployment dependencies
- `streamlit_app/titanic_pipeline.pkl` — the trained model pipeline from Week 4
- `README.md` — this document

### Next Steps

With a working deployed model in place, future iterations could extend the app to expose the churn model as well, and explore whether an ensemble model from Week 4 (Random Forest or XGBoost) improves live prediction quality over the current Logistic Regression pipeline.

---
*Part of the NeuroFive Solutions ML Track — Week 5: Machine Learning Fundamentals*
