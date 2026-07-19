# NeuroFive ML Track — Week 2

## Task 1: Predict Titanic Survival — Classification Model

### Overview

Building on the exploratory analysis from Week 1, this task focused on training my first machine learning model: a Logistic Regression classifier to predict passenger survival on the Titanic. This is a binary classification problem, where the goal is to assign each passenger to one of two categories — survived or did not survive — based on features like class, sex, age, and fare.

### Approach

I started from the cleaned dataset produced during Week 1's EDA, addressing the remaining data quality issues before modeling. Missing values in `Age` were filled with the median, missing values in `Embarked` were filled with the mode, and `Cabin` was dropped entirely given how sparse it was. I also removed `PassengerId`, `Name`, and `Ticket`, since none of these carry direct predictive value in their raw form.

The categorical features `Sex` and `Embarked` were one-hot encoded using `pd.get_dummies()`, converting them into a numerical format the model could use, while dropping the first category of each to avoid redundant columns.

With the data prepared, I split it into training and test sets using an 80/20 split via `train_test_split`, with a fixed random state for reproducibility. I trained a Logistic Regression model on the training set and evaluated it against the held-out test set, which contained 179 passengers the model had never seen.

### Results

The model achieved an accuracy of **81.01%** on the test set.

The confusion matrix breaks this down further:

|                     | Predicted: Did Not Survive | Predicted: Survived |
|---------------------|:---------------------------:|:---------------------:|
| **Actual: Did Not Survive** | 90 (True Negative) | 15 (False Positive) |
| **Actual: Survived**        | 19 (False Negative) | 55 (True Positive) |

Out of 179 test passengers, the model correctly classified 145 (90 + 55), and misclassified 34. It performed slightly better at identifying passengers who did not survive (86% recall) than those who did (74% recall), meaning the model is somewhat more likely to miss an actual survivor than to miss an actual non-survivor. This is reflected in the classification report: precision and recall for the "Did Not Survive" class were 0.83 and 0.86 respectively, compared to 0.79 and 0.74 for "Survived."

Overall, with a weighted F1-score of 0.81, the model performs reasonably well as a first baseline, though there is clear room for improvement — particularly in catching more true survivors.

---

## Task 2: House Price Prediction — Linear Regression

### Overview

This task introduced regression, used to predict a continuous numerical value rather than a category. I trained a Linear Regression model on the California Housing dataset (`sklearn.datasets`) to predict median house value.

### Approach

I selected four features believed to most affect price: median income (`MedInc`), house age (`HouseAge`), average rooms (`AveRooms`), and average occupancy (`AveOccup`). After an 80/20 train/test split, I trained a Linear Regression model and evaluated it using RMSE (Root Mean Squared Error) and R² score. I also plotted predicted versus actual prices to visually assess the model's quality, with points close to the diagonal reference line indicating accurate predictions.

### Results

*[RMSE, R² score, and the plain-English R² explanation to be added once final values are confirmed.]*

### Tools and Libraries

- Python 3.10.5, Visual Studio Code with the Jupyter extension
- pandas, numpy, scikit-learn, matplotlib

### Repository Contents

- `Week2_Classification_model.ipynb` — Titanic survival classification
- `Week2_Regression_model.ipynb` — California housing price regression
- `README.md` — this document

### Next Steps

For the classification task, I plan to explore hyperparameter tuning and additional evaluation metrics beyond accuracy in Week 3. For the regression task, I intend to test additional features and compare Linear Regression against other regression models to see whether R² can be improved.

---
*Part of the NeuroFive Solutions ML Track — Week 2: Machine Learning Fundamentals*
