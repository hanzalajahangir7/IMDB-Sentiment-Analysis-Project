# IMDB-Sentiment-Analysis-Project
IMDB Sentiment Analysis using Keras. This project builds a neural network to classify movie reviews as positive or negative by vectorizing text data with multi-hot encoding and TF-IDF. It includes model training with dropout, evaluation, loss visualization, and saving the model &amp; vectorizer for future use.

IMDB Movie Review Sentiment Analysis

This project implements a binary sentiment classification model to determine whether a movie review from the IMDB dataset is positive or negative. Using TensorFlow and Keras, it leverages a fully connected neural network trained on vectorized text data.

Features

Data Loading: Uses the IMDB dataset from Keras, restricted to the top 10,000 most frequent words.

Data Preprocessing: Text reviews are converted to multi-hot encoded vectors (binary word presence) to serve as input features.

Model Architecture: A sequential neural network with two hidden layers (16 units each) and dropout (50%) to reduce overfitting.

Training: Model trained for 20 epochs with RMSprop optimizer and binary cross-entropy loss function.

Validation: A validation split of 10,000 samples is used to monitor overfitting during training.

Evaluation: Model performance evaluated on the test set achieving around 88% accuracy.

Visualization: Training and validation loss plotted to visualize learning progress and overfitting.

Model Saving: The trained model is saved as an .h5 file.

Vectorizer Saving: TF-IDF vectorizer fitted on raw decoded reviews is saved for future text preprocessing consistency.

What I Achieved

Successfully built a deep learning model for sentiment analysis on a standard NLP dataset.

Demonstrated effective preprocessing techniques (multi-hot encoding & TF-IDF).

Incorporated dropout layers to mitigate overfitting.

Achieved a strong validation accuracy (~88-89%).

Provided reproducible model and vectorizer saving for deployment or further experimentation.

Potential Improvements

Experiment with embedding layers instead of multi-hot encoding for richer word representations.

Use more advanced architectures such as LSTM or Transformer models for sequential context.

Tune hyperparameters like learning rate, dropout rate, and batch size for improved accuracy.

Expand dataset preprocessing by handling padding or truncation for better input consistency.

Add unit tests and scripts for easier deployment or batch prediction.
