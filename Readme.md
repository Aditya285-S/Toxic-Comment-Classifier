# Toxic Comment Classifier

This project is application built using Streamlit to classify toxic comments. The classifier uses TF-IDF (Term Frequency-Inverse Document Frequency) for text feature extraction and a Random Forest Classifier for prediction. The app allows users to input text or upload an image from which text is extracted for classification.

## Overview

The Toxic Comment Classifier is designed to identify and classify toxic comments based on their content. It employs machine learning techniques to analyze text input and provide predictions on whether a comment is toxic or not.

## Features

- Users can enter text directly into the app for classification.
- Alternatively, users can upload an image containing text, and the classifier will extract the text for analysis.
- The app predit the probability of toxicityin the input text.

## Dataset

The classifier model is trained on a dataset of labeled comments. The dataset consists of text comments labeled with toxicity categories (e.g., toxic, threat, hate,insult,etc). This labeled data is used to train the TF-IDF vectorizer and the Random Forest Classifier.

## Algorithm Used

### TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF is a popular technique used to convert text documents into numerical feature vectors. It evaluates the importance of a word in a document relative to a collection of documents (corpus). The TF-IDF value increases proportionally to the number of times a word appears in a document and is offset by the frequency of the word in the corpus.

### Random Forest Classifier

Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes (classification) or the mean prediction (regression) of the individual trees. It is robust against overfitting and performs well in handling high-dimensional data like text features.


## Images of Visualizations

The app includes visualizations to provide insights into the classifier's performance and the dataset characteristics. Some example visualizations are:

### Toxic comments counts

![](https://github.com/Aditya285-S/Toxic-Comment-Classifier/blob/main/Visulizations/comment%20count.png)

### Word Cloud of Toxic comment

![](https://github.com/Aditya285-S/Toxic-Comment-Classifier/blob/main/Visulizations/World%20clound%20of%20toxic%20comments.png)


## Usage

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt

