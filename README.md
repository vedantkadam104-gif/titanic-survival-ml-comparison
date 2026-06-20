# Titanic Survival Prediction - ML Model Comparison

Project 02 for the Pluto Academy AI & ML Internship.

## About this project

This project tries to predict whether a Titanic passenger survived or not, using the classic Titanic dataset from Kaggle. The main goal was to train a few different ML models, compare how well they perform, and figure out which one works best and why.

Colab Link - https://colab.research.google.com/drive/1PbiL4hq_TNcJrYrJYnziq9D0Vcf6gT2V?usp=sharing

Dataset: https://www.kaggle.com/c/titanic (891 passengers, 12 columns)

## Tools used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Google Colab

## What I did

**Cleaning the data**
- Age had 177 missing values, so I filled them using the median age within each Pclass + Sex group instead of just one overall number
- Embarked had only 2 missing values, filled with the most common port
- Cabin was missing for 77% of rows, so instead of dropping it completely I made a new column called HasCabin (1 if known, 0 if not)
- Dropped PassengerId and Ticket since they don't help predict survival

**Feature engineering**
- Pulled out a Title (Mr/Mrs/Miss/Rare) from the Name column
- Converted Sex to 0/1
- One-hot encoded Embarked and Title

**Models trained**
Logistic Regression, Random Forest, and KNN. Used an 80/20 train-test split. Scaled the data before feeding it to Logistic Regression and KNN since they're sensitive to feature scale, didn't scale for Random Forest since tree models don't need it.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.832 | 0.797 | 0.797 | 0.797 |
| Random Forest | 0.832 | 0.797 | 0.797 | 0.797 |
| KNN | 0.799 | 0.757 | 0.757 | 0.757 |

Logistic Regression and Random Forest ended up tied. I went with Random Forest as the best model since it also gives feature importance, which helps explain the predictions instead of just giving a number.

Confusion matrix for Random Forest: 90 true negatives, 15 false positives, 15 false negatives, 59 true positives.

Top features by importance: Fare, Age, Title_Mr, Sex.

## Conclusion

Random Forest got 83.2% accuracy with precision and recall both at 79.7%, and an equal number of false positives and false negatives, so it's not biased toward predicting one outcome over the other. Fare, Age, and Sex/Title being the top features lines up with what's known historically about the disaster - wealthier and female passengers had a better chance of survival. KNN did slightly worse (79.9%), probably because it's more affected by the mix of scaled numeric and one-hot encoded columns.

## How to run this

1. Clone this repo
2. Open the notebook in Google Colab
3. Get train.csv from the Kaggle Titanic competition page and upload it (or mount Drive)
4. Run all cells top to bottom

## Author

Vedant Kadam
M.Sc. Data Science & AI, Savitribai Phule Pune University
