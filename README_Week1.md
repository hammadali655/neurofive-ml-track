# NeuroFive ML Track — Week 1

## Set Up Your Data Science Toolkit & Explore Your First Dataset

### Overview

This repository documents my first task in the NeuroFive Solutions Machine Learning track. Before writing any modeling code, I focused on two fundamentals every data scientist relies on: a reliable local development environment, and the discipline of understanding a dataset thoroughly before working with it.

### What I Did

I began by setting up a Python virtual environment in VS Code to keep this project's dependencies isolated and reproducible. Within that environment, I installed the core data science toolkit — `pandas` and `numpy` for data handling, `ipykernel` to run Jupyter notebooks natively in VS Code, and `matplotlib` and `seaborn` for visualization. Along the way, I ran into a kernel configuration issue where VS Code was defaulting to a global Python interpreter instead of my virtual environment; I resolved this by explicitly registering and selecting a dedicated kernel for the project.

With the environment in place, I downloaded the Titanic dataset from Kaggle's "Titanic - Machine Learning from Disaster" competition and loaded it using `pandas.read_csv()`. I then worked through a structured exploratory data analysis: inspecting the dataset's shape, reviewing column data types, checking for missing values, and distinguishing categorical from numerical features. To make these findings easier to interpret, I built a series of visualizations covering missing data patterns, survival distribution, and relationships between survival and features such as passenger class, sex, age, and fare.

Finally, I documented my observations in a short written summary within the notebook, treating this first analysis as a starting point I can build on in later weeks.

### Key Findings

The dataset contains 891 rows and 12 columns. Three columns have missing values: `Age` (roughly 20% missing), `Cabin` (over 75% missing), and `Embarked` (2 missing values). The dataset splits into five categorical columns — `Name`, `Sex`, `Ticket`, `Cabin`, and `Embarked` — and seven numerical columns, including the target variable, `Survived`.

The visualizations surfaced a few patterns worth noting. Survival rates were noticeably higher among women and among first-class passengers, which aligns with historical accounts of the disaster. The `Fare` column is right-skewed, with a small number of high-value outliers concentrated in first class. I also noted that `Pclass`, while stored as an integer, functions as a categorical variable and should likely be treated as such in any future modeling work.

### Tools and Libraries

- Python 3.10.5
- Visual Studio Code with the Jupyter extension
- pandas, numpy, matplotlib, seaborn

### Repository Contents

- `titanic-eda.ipynb` — the complete exploratory analysis, including code, visual outputs, and written observations
- `train.csv` — the Titanic dataset sourced from Kaggle
- `README.md` — this document

### Next Steps

Going forward, I plan to address the missing values in `Age` and determine an appropriate strategy for handling `Cabin`, given how sparse it is. I also intend to engineer additional features — such as extracting titles from passenger names and calculating family size from `SibSp` and `Parch` — as preparation for baseline model training in Week 2.

---
*Part of the NeuroFive Solutions ML Track — Week 1: Machine Learning Fundamentals*
