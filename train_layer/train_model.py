from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import json
import pickle
import logging

def get_data(input_path):
    with open (input_path, mode='r') as f:
        data = json.load(f)
    return data


def train_model(training_data):

    X_train = [i['text'] for i in train_data]
    y_train = [i['label'] for i in train_data]

    vectorizer = CountVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)

    model = MultinomialNB()

    model.fit(X_train_vectors, y_train)

    return model


def test_model(model, test_data):

    X_test = [i['text'] for i in train_data]
    y_test = [i['label'] for i in train_data]
    
    vectorizer = CountVectorizer()
    X_test_vectors = vectorizer.fit_transform(X_test)

    y_pred = model.predict(X_test_vectors)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    confusion = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, labels=[0,1, 2, 3, 4, 5])

    print(f'Accuracy score: {accuracy}')
    print(f'Precision score: {precision}')
    print(f'Recall score: {recall}')
    print(f'F1 score: {f1}')

    logger = logging.getLogger(__name__)
    logging.basicConfig(filename='./model_logger.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s: %(levelname)s: %(message)s')  
    logger.info(f'Model has been tested with data.\nConfusion Matrix: \n{confusion}\n Classification Report: \n{class_report}\n')

    print('The confusion matrix and classification report for the tested data has been logged.')


def save_trained_model(model):
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print('Trained model has been saved.')


train_data_path = './ingest_layer/emotion_train.json'
train_data = get_data(train_data_path)

model = train_model(train_data)

test_data_path = './ingest_layer/emotion_test.json'
test_data = get_data(test_data_path)

test_model(model, test_data)

validation_data_path = './ingest_layer/emotion_validation.json'
validation_data = get_data(validation_data_path)
test_model(model, validation_data)

save_trained_model(model)