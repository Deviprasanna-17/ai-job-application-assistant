# Fake Job Posting Detection using Machine Learning

An NLP-based machine learning tool that detects fraudulent job postings, built as part of an end-to-end job application assistant project.

## Problem
Fake job postings are a real, growing scam risk for job seekers. This project detects fraudulent postings using text analysis and machine learning, so applicants can screen listings before applying.

## Dataset
[Real / Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) from Kaggle — 17,880 job postings, 866 labeled as fraudulent.

## Approach
1. **Data Cleaning**: Removed HTML artifacts, fixed word-jamming issues, combined description/requirements/company profile into one text field
2. **Feature Engineering**: TF-IDF vectorization (5,000 features)
3. **Handling Class Imbalance**: Applied SMOTE (Synthetic Minority Oversampling) to balance the training data, since only ~5% of postings are fraudulent
4. **Model**: Random Forest Classifier, trained on SMOTE-balanced data
5. **Threshold Tuning**: Tuned the decision threshold to 0.4 to balance precision and recall for the fraud class

## Results
| Metric | Score |
|---|---|
| Precision (fraud) | 0.86 |
| Recall (fraud) | 0.71 |
| F1-score (fraud) | 0.77 |

## Tools Used
Python, pandas, scikit-learn, imbalanced-learn (SMOTE), Jupyter/Colab
