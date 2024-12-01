[Name: Saishree Mishra 

Company: CODTECH IT SOLUTIONS 

ID: CT08DS10056 

Domain: Machine Learning 

Duration: November 15 to December 15 2024

Mentor: Neela Santhosh Kumar]




Overview of the Task

TASK: Sentiment Analysis on IMDB Dataset Using Naive Bayes.
The goal of this task is to perform sentiment analysis on movie reviews from the IMDB dataset. By training a Naive Bayes model, we aim to classify reviews as either positive or negative based on the text content.

Objective:-

To predict the sentiment (positive/negative) of movie reviews using a text classification model. This project emphasizes preprocessing, feature extraction, model training, evaluation, and prediction.


Key Activities:-

1. Data Preparation
Dataset: IMDB Dataset is loaded from the specified file path (IMDB Dataset.csv).
Initial Exploration:
Display the first few rows of the dataset to understand its structure.
Check data types and completeness using data.info() to identify null or missing values.
Target Encoding: Sentiments are mapped to numeric values: positive to 1 and negative to 0.

2. Feature Engineering
Text Data to Numeric Conversion:
Used CountVectorizer from sklearn.feature_extraction.text to convert text reviews into numerical feature vectors.
Configuration:
stop_words='english': Removes common English stop words.
max_features=5000: Limits the vocabulary size to the 5000 most frequent terms.
Training/Testing Transformation:
The training dataset is fitted and transformed into a sparse matrix representation.
The test dataset is transformed using the same vocabulary.

3. Model Implementation
Model: MultinomialNB (Multinomial Naive Bayes)
Suitable for text classification tasks where features represent word counts or frequencies.
Training: The model is trained on the vectorized training data (X_train_vect) and corresponding labels (y_train).

4. Model Evaluation
Predictions:
Sentiment predictions are made for the test dataset (X_test_vect).
Performance Metrics:
Accuracy Score: Proportion of correctly classified reviews.
Classification Report: Includes precision, recall, and F1-score for each class (positive/negative).
Example Prediction: Demonstrates the model’s output for a user-provided review.

5. Example Prediction
An example review is vectorized using the trained CountVectorizer and passed to the model for sentiment classification. The result (positive/negative sentiment) is displayed.


Technologies Used:-

Python Libraries:
pandas: For data loading and manipulation.
numpy: For numerical computations.
scikit-learn:
(CountVectorizer: To convert text to numerical data.
MultinomialNB: For training the Naive Bayes classifier.
Metrics: For evaluating model performance.)

Dataset: IMDB Dataset (IMDB Dataset.csv).

Evaluation Metrics
(Accuracy: Measures overall model performance.
Classification Report:
Precision: Correctly identified positives/total predicted positives.
Recall: Correctly identified positives/total actual positives.
F1-Score: Harmonic mean of precision and recall.)


Visualization:-
Though not explicitly implemented in this code, visualizations like confusion matrices and word importance charts could be added to enhance interpretability.

