# Customer Churn Prediction

## Problem
Predicting whether a telecom customer will churn using machine learning.

## Dataset
- Source: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- 7032 rows, 21 features
- Class imbalance: 73% No, 27% Yes

```
customer-churn/
├── Data/
│   └── Telco-Customer-Churn.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_models.ipynb
├── src/
│   ├── preprocess.py
│   └── train.py
├── requirements.txt
└── README.md
```


## Models & Results
| Model | Accuracy | Churn F1 |
|-------|----------|----------|
| Logistic Regression | 0.73 | 0.61 |
| Decision Tree | 0.69 | 0.59 |
| XGBoost | 0.74 | 0.60 |
| XGBoost (tuned) | 0.73 | 0.61 |

## Key Findings
- Contract type is the most important feature
- Fiber optic internet users churn more
- Class imbalance handled with scale_pos_weight and class_weight

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook
```