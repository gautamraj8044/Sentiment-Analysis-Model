
# IMDb Sentiment Analysis

This project focuses on building a sentiment analysis model using IMDb movie reviews. The model classifies reviews as positive or negative based on the text content.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to analyze and predict the sentiment of movie reviews from the IMDb dataset. By preprocessing the text data and applying machine learning techniques, we aim to create a robust model that can accurately classify reviews as positive or negative.

## Dataset
The dataset used in this project is the IMDb movie reviews dataset. It contains two columns:
- `review`: The text of the movie review.
- `sentiment`: The sentiment of the review, either "positive" or "negative".

## Data Preprocessing
1. **Text Cleaning**: Removed HTML tags and unnecessary punctuation.
2. **Tokenization & Lemmatization**: Tokenized the text and reduced words to their base forms using spaCy.
3. **Stop Word Removal**: Filtered out common stop words to focus on meaningful words.

## Model
The model uses a Logistic Regression classifier. Text data is vectorized using `CountVectorizer` to convert the cleaned text into a numerical format suitable for machine learning. The dataset is split into training and testing sets to evaluate the model's performance.

## Installation
To run this project, you need to have Python installed. Clone this repository and install the required packages using the following commands:

```bash
git clone https://github.com/gautamraj8044/Sentiment-Analysis-Model
cd imdb-sentiment-analysis
pip install -r requirements.txt
```

The `requirements.txt` file should contain:
```
pandas
spacy
scikit-learn
```

## Usage
1. **Preprocess the data**: Clean and preprocess the IMDb reviews.
2. **Train the model**: Train the Logistic Regression model on the preprocessed data.
3. **Evaluate the model**: Test the model on the test set and evaluate its performance.
4. **Predict sentiment**: Use the trained model to predict the sentiment of new reviews.

```python
from sentiment_analysis import predict_sentiment

new_review = "I didn't like the film at all. It was boring and predictable."
predicted_sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {predicted_sentiment}")
```

## Results
The model achieves good accuracy in predicting the sentiment of the reviews. Detailed results and metrics, such as precision, recall, and F1-score, can be found in the output of the `classification_report`.

## Future Work
- Improve text preprocessing by exploring different techniques.
- Experiment with other machine learning models and feature extraction methods.
- Incorporate more advanced NLP techniques, such as word embeddings and deep learning models.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any improvements or bug fixes.

---