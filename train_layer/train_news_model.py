from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import json
import pickle
import logging

logging.basicConfig(filename='./logging/news_model_logger.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s: %(levelname)s: %(message)s') 

class TrainModel:
    def __init__(self):
        self.vectorizer = CountVectorizer()
        # self.model = RandomForestClassifier(n_estimators=20)
        self.model = MultinomialNB()
        self.logger = logging.getLogger(__name__)
    

    def get_data(self, input_path):
        with open (input_path, mode='r') as f:
            data = json.load(f)
        X = [d['title'] for d in data]
        y = [d['topic'] for d in data]
        
        RANDOM_SEED = 1276
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
        
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):

        X_train_vectors = self.vectorizer.fit_transform(X_train)

        self.model.fit(X_train_vectors, y_train)

        return self.model


    def test_model(self, X_test, y_test):
        
        X_test_vectors = self.vectorizer.transform(X_test)

        y_pred = self.model.predict(X_test_vectors)

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

        self.logger.info(f'Model has been tested with data.\nConfusion Matrix: \n{confusion}\n Classification Report: \n{class_report}\n')

        print('The confusion matrix and classification report for the tested data has been logged.')


    def save_trained_model(self):
        with open('./models/trained_classifier_model.pkl', 'wb') as f:
            pickle.dump({'model':self.model, 'vectorizer': self.vectorizer}, f)
        print('Trained model has been saved.')



train_news_model = TrainModel()

input_path = 'ingest_layer/cleaned_labeled_newscatcher_dataset.json'
X_train, X_test, y_train, y_test = train_news_model.get_data(input_path)

trained_model = train_news_model.train_model(X_train, y_train)

train_news_model.test_model(X_test, y_test)

train_news_model.save_trained_model()