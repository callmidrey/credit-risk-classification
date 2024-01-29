# credit-risk-classification
This is my module 20 Challenge

# - Purpose of the Analysis:
The primary objective was to build machine learning models capable of predicting loan risk, distinguishing between healthy and high-risk loans based on financial data.

# - Financial Information and Prediction Target:
The dataset contained financial information related to loan applications, including attributes such as loan size,borrower income, debt-to-income, number of accounts,deregatory marks, total debt and loan status.
The prediction target was the loan status, with two classes: healthy loans (0) and high-risk loans (1).

# - Variable Information:
To gain insights into the distribution of loan status, value_counts() was used to examine the frequency of healthy and high-risk loans.
For example, there might be 18,765 instances of healthy loans (0) and 619 instances of high-risk loans (1).


# Stages of the Machine Learning Process:
- Data Preprocessing: 
The data was a good one with almost no missing values and it was a complete dataset. I just had to load the csv file properly through the appropriate path and it was set to go.
- Model Selection: The classification algorithm, Logistic Regression, was required to be used for this assignment and considered as the most suitable model for the task.
- Model Training: The selected model was trained on both the orignal dataset and the predicted preprocessed dataset to learn patterns and relationships between features and the target variable.
- Model Evaluation: The trained models were evaluated using metrics such as accuracy, precision, recall, and F1-score to assess their performance in predicting loan risk.


# - Methods Used:
- Logistic Regression: Logistic regression is primarily used for binary classification problems, where the target variable (dependent variable) has only two possible outcomes or classes, typically represented as 0 and 1 which is a common choice for binary classification tasks like loan risk assessment found in this dataset.

- Resampling Methods: Resampling methods are techniques used in machine learning to address imbalances in the distribution of classes within a dataset. When dealing with imbalanced datasets, where one class is significantly more prevalent than the others, traditional machine learning algorithms may struggle to effectively learn from and make accurate predictions on such data. Random Oversampling was employed to address class imbalance issues, ensuring that the model is trained on a balanced dataset representative of both healthy and high-risk loans.

In summary, the analysis involved preparing the data, selecting and training machine learning models, evaluating their performance, and employing appropriate techniques to address class imbalance. The ultimate goal was to develop accurate and reliable models for predicting loan risk, aiding financial institutions in making informed lending decisions.

## Results
# keys 
'Accuracy: Accuracy measures the overall correctness of the model's predictions.' 
`Precision: Precision measures the accuracy of positive predictions.`
`Recall: Recall, also known as sensitivity, measures the proportion of actual positives that were correctly identified by the model.`

* Machine Learning Model 1 (Orignal data Logistic Model):
![Alt text](<Screenshot 2024-01-29 011521.png>)
- Accuracy: The accuracy score of 0.99 indicates that the model correctly predicts the loan risk category for 99% of the instances in the dataset. This high accuracy suggests that the model's predictions align well with the actual loan risk labels.
- Precision: For the healthy loan category (0), the precision score of 1.00 indicates that all loans predicted as healthy are indeed healthy. For the high-risk loan category (1), the precision score of 0.84 indicates that 84% of the loans predicted as high-risk are indeed high-risk. In practical terms, this means that while the model is highly accurate in identifying healthy loans, it may have some false positives when predicting high-risk loans.
- Recall: The recall score of 0.99 for healthy loans (0) suggests that the model captures 99% of all actual healthy loans. For high-risk loans (1), the recall score of 0.94 indicates that the model identifies 94% of all actual high-risk loans. This implies that the model effectively identifies the majority of high-risk loans, with a slightly lower recall compared to healthy loans.


* Machine Learning Model 2 (Predicted data Logistic Model):
![Alt text](<Screenshot 2024-01-29 011536.png>)
- Accuracy: The accuracy score of 0.99 indicates that the model correctly predicts the loan risk category for 99% of the instances in the dataset. This high accuracy suggests that the model's predictions align well with the actual loan risk labels.
- Precision: For the healthy loan category (0), the precision score of 1.00 indicates that all loans predicted as healthy are indeed healthy. For the high-risk loan category (1), the precision score of 0.84 indicates that 84% of the loans predicted as high-risk are indeed high-risk. In practical terms, this means that while the model is highly accurate in identifying healthy loans, it may have some false positives when predicting high-risk loans.
- Recall: The recall score of 0.99 for healthy loans (0) suggests that the model captures 99% of all actual healthy loans. For high-risk loans (1), the recall score of 0.99 indicates that the model identifies 99% of all actual high-risk loans. This implies that the model effectively identifies the majority of high-risk loans, with an equal recall compared to healthy loans.


## Summary
Based on the results of the Logistic Regression models, the Machine Learning Model 2 (Predicted data Logistic Model) seems to perform better overall compared to the Machine Learning Model 1 (Orignal data Logistic Model):

Performance Evaluation: Both models have an accuracy score (0.99) which is very good. Additionally, the precision and recall scores for both classes are the same, indicating equal performance for both healthy (Class 0) and high-risk (Class 1) loans.

Importance of Predictions: The importance of predictions depends on the problem we are trying to solve. In the context of loan risk assessment, it is crucial to predict both healthy (Class 0) and high-risk (Class 1) loans accurately. However, the consequences of misclassifications may vary. For instance, misclassifying a healthy loan as high-risk (false positive) may result in missed lending opportunities, while misclassifying a high-risk loan as healthy (false negative) may lead to increased risk for the lending institution.
Considering the higher accuracy and better precision-recall trade-offs, the Machine Learning Model 2 (Predicted data Logistic Model) appears to be more suitable for predicting loan risk categories as it has a better score (0.99) than the score of (0.94) of the Machine Learning Model 1 (Orignal data Logistic Model). 

# Conclusion:
Bothe Models are good. However, it's essential to consider the specific requirements and priorities of the lending institution when selecting the most appropriate model for deployment. If the emphasis is on minimizing false positives (predicting high-risk loans accurately), the Machine Learning Model 2 (Predicted data Logistic Model) would be the preferred choice.


