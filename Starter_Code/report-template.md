# Module 12 Report Template

## Overview of the Analysis

# - Purpose of the Analysis:
The primary objective was to build machine learning models capable of predicting loan risk, distinguishing between healthy and high-risk loans based on financial data.

# - Financial Information and Prediction Target:
The dataset contained financial information related to loan applications, including attributes such as credit scores, income levels, debt-to-income ratios, and loan amounts.
The prediction target was the loan risk category, with two classes: healthy loans (0) and high-risk loans (1).

# - Variable Information:
To gain insights into the distribution of loan categories, value_counts() was used to examine the frequency of healthy and high-risk loans.
For example, there might be 18,765 instances of healthy loans (0) and 619 instances of high-risk loans (1).

# Stages of the Machine Learning Process:
- Data Preprocessing: 
This involved handling missing values, encoding categorical variables, and scaling numerical features to prepare the data for modeling.

Model Selection: Various classification algorithms such as Logistic Regression, Random Forest, and Support Vector Machines (SVM) may have been considered to identify the most suitable model for the task.

Model Training: The selected model(s) were trained on the preprocessed dataset to learn patterns and relationships between features and the target variable.

Model Evaluation: The trained models were evaluated using metrics such as accuracy, precision, recall, and F1-score to assess their performance in predicting loan risk.


# - Methods Used:
Logistic Regression: A common choice for binary classification tasks like loan risk assessment.
Resampling Methods: Techniques such as Random Oversampling were employed to address class imbalance issues, ensuring that the model is trained on a balanced dataset representative of both healthy and high-risk loans.

In summary, the analysis involved preparing the data, selecting and training machine learning models, evaluating their performance, and employing appropriate techniques to address class imbalance. The ultimate goal was to develop accurate and reliable models for predicting loan risk, aiding financial institutions in making informed lending decisions.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
- Accuracy: The accuracy score of 0.99 indicates that the model correctly predicts the loan risk category for 99% of the instances in the dataset. This high accuracy suggests that the model's predictions align well with the actual loan risk labels.
- Precision: For the healthy loan category (0), the precision score of 1.00 indicates that all loans predicted as healthy are indeed healthy. For the high-risk loan category (1), the precision score of 0.84 indicates that 84% of the loans predicted as high-risk are indeed high-risk. In practical terms, this means that while the model is highly accurate in identifying healthy loans, it may have some false positives when predicting high-risk loans.
- Recall: The recall score of 0.99 for healthy loans (0) suggests that the model captures 99% of all actual healthy loans. For high-risk loans (1), the recall score of 0.94 indicates that the model identifies 94% of all actual high-risk loans. This implies that the model effectively identifies the majority of high-risk loans, with a slightly lower recall compared to healthy loans.


* Machine Learning Model 2:
- Accuracy: Accuracy measures the overall correctness of the model's predictions. An accuracy of 0.99 suggests that the model correctly predicts the class label for 99% of the instances in the dataset.
- Precision: Precision measures the accuracy of positive predictions. A precision score of 1.00 for class 0 and 0.84 for class 1 indicates that the model accurately identifies the majority of true positives for class 0 and a slightly lower proportion for class 1.
- Recall: Recall, also known as sensitivity, measures the proportion of actual positives that were correctly identified by the model. A recall score of 0.99 for class 0 and 0.99 for class 1 suggests that the model effectively captures the majority of positive instances for both classes.


## Summary
Based on the results of the Logistic Regression models, the predicted data logistic regression model seems to perform better overall compared to the orignal data logistic regression model:

Performance Evaluation: The predicted data Logistic Regression model has a higher accuracy score (0.99) compared to the Random Forest model (0.98). Additionally, the precision and recall scores for both classes are generally higher in the Logistic Regression model, indicating better predictive performance for both healthy (Class 0) and high-risk (Class 1) loans.

Importance of Predictions: The importance of predictions depends on the problem we are trying to solve. In the context of loan risk assessment, it is crucial to predict both healthy (Class 0) and high-risk (Class 1) loans accurately. However, the consequences of misclassifications may vary. For instance, misclassifying a healthy loan as high-risk (false positive) may result in missed lending opportunities, while misclassifying a high-risk loan as healthy (false negative) may lead to increased risk for the lending institution.

Considering the higher accuracy and better precision-recall trade-offs, the Logistic Regression model appears to be more suitable for predicting loan risk categories. However, it's essential to consider the specific requirements and priorities of the lending institution when selecting the most appropriate model for deployment. If the emphasis is on minimizing false positives (predicting high-risk loans accurately), the Logistic Regression model would be the preferred choice.

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
