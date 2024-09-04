import os
import re
import nltk
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import random
from sklearn.metrics import accuracy_score,f1_score
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# def print_confusion_matrix(test_labels, test_predictions):
#     cm = confusion_matrix(test_labels, test_predictions)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=sentiment_categories, yticklabels=sentiment_categories)
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     plt.title('Confusion Matrix')
#     plt.show()


def load_reviews_from_file(file_path):
    with open(file_path, 'r') as f:
        return f.read().splitlines()


path_to_sentiment_data = 'sentiment_data'


sentiment_categories = ['Excellent', 'Good', 'Average', 'Bad', 'Poor']


features_data = []


for category in sentiment_categories:
    category_file_path = os.path.join(path_to_sentiment_data, category.lower() + '.txt')
    if os.path.exists(category_file_path):
        features_data.extend([(review, category) for review in load_reviews_from_file(category_file_path)])
    else:
        print(f"Warning: File not found for category '{category}'")


random.shuffle(features_data)


threshold = 0.8
num_samples = len(features_data)
num_train = int(threshold * num_samples)


train_reviews, train_labels = zip(*features_data[:num_train])
test_reviews, test_labels = zip(*features_data[num_train:])


with open('notrequired.txt', 'r') as f:
    unnecessary_words = set(f.read().split())

    

def custom_tokenizer(text, unnecessary_words):

    tokens = word_tokenize(text)

    modified_tokens = []

    for i, token in enumerate(tokens):

        if token.lower() not in unnecessary_words and token.isalnum() and not re.match(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', token):
            modified_tokens.append(token.lower())

    return modified_tokens





def custom_tokenizer_ngrams(text, unnecessary_words):
    return list(nltk.ngrams(custom_tokenizer(text, unnecessary_words), 1)) + \
           list(nltk.ngrams(custom_tokenizer(text, unnecessary_words), 2)) + \
           list(nltk.ngrams(custom_tokenizer(text, unnecessary_words), 3))


svm_classifier = make_pipeline(
    TfidfVectorizer(tokenizer=lambda text: custom_tokenizer_ngrams(text, unnecessary_words), max_features=5000),
    SVC(kernel='rbf')
)


svm_classifier.fit(train_reviews, train_labels)

def predict_sentiment(text):
    sentences = []
    sentence = ''
    for word in text.split():
        sentence += word + ' '
        if word in [',', '.', 'and', 'but']:
            sentences.append(sentence.strip())
            sentence = ''
    if sentence:
        sentences.append(sentence.strip())
   
    sentence_sentiments = []
    sentence_scores = []
    

    sentiment_score_mapping = {
        "Excellent": 2,
        "Good": 1,
        "Average": 0,
        "Bad": -1,
        "Poor": -2
    }

    for sentence in sentences:

        sentiment = svm_classifier.predict([sentence])[0]
        sentence_sentiments.append(sentiment)
        sentence_scores.append(sentiment_score_mapping.get(sentiment, 0))
  
    if sentence_scores:
        average_score = sum(sentence_scores) / len(sentence_scores)
    else:
        average_score = 0
    

    if average_score >= 1.5:
        overall_sentiment = "Excellent"
    elif average_score >= 0.5:
        overall_sentiment = "Good"
    elif average_score >= -0.5:
        overall_sentiment = "Average"
    elif average_score >= -1.5:
        overall_sentiment = "Bad"
    else:
        overall_sentiment = "Poor"
    

    emoji_mapping = {
        "Excellent": "😊 Excellent",
        "Good": "😄 Good",
        "Average": "😐 Average",
        "Bad": "😞 Bad",
        "Poor": "😢 Poor"
    }

    return emoji_mapping.get(overall_sentiment, "Unknown")


if __name__ == "_main_":
    test_text = "I love this movie! It's so good. However, the ending was a bit disappointing."
    print(predict_sentiment(test_text))

test_predictions = svm_classifier.predict(test_reviews)
accuracy = accuracy_score(test_labels, test_predictions)
f1 = f1_score(test_labels, test_predictions, average='weighted')
print("F1 Score:", f1)
print("Accuracy:", accuracy)
# print_confusion_matrix(test_labels, test_predictions)