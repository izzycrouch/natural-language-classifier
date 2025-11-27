
'''Tested first with a naive bayes model,  but random forest model provided more accurate results. '''

# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
# import json
# import pickle
# import logging

# logging.basicConfig(filename='./logging/NB_model_logger.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s: %(levelname)s: %(message)s') 

# class TrainNBModel:
#     def __init__(self):
#         self.vectorizer = CountVectorizer()
#         self.model = MultinomialNB()
#         self.logger = logging.getLogger(__name__)
    

#     def get_data(self, input_path):
#         with open (input_path, mode='r') as f:
#             data = json.load(f)
#         return data


#     def train_model(self, training_data):

#         X_train = [i['text'] for i in training_data]
#         y_train = [i['label'] for i in training_data]

#         X_train_vectors = self.vectorizer.fit_transform(X_train)

#         self.model.fit(X_train_vectors, y_train)

#         return self.model


#     def test_model(self, test_data):

#         X_test = [i['text'] for i in test_data]
#         y_test = [i['label'] for i in test_data]
        
#         X_test_vectors = self.vectorizer.transform(X_test)

#         y_pred = self.model.predict(X_test_vectors)

#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred, average='weighted')
#         recall = recall_score(y_test, y_pred, average='weighted')
#         f1 = f1_score(y_test, y_pred, average='weighted')
#         confusion = confusion_matrix(y_test, y_pred)
#         class_report = classification_report(y_test, y_pred, labels=[0,1, 2, 3, 4, 5])

#         print(f'Accuracy score: {accuracy}')
#         print(f'Precision score: {precision}')
#         print(f'Recall score: {recall}')
#         print(f'F1 score: {f1}')

#         self.logger.info(f'Model has been tested with data.\nConfusion Matrix: \n{confusion}\n Classification Report: \n{class_report}\n')

#         print('The confusion matrix and classification report for the tested data has been logged.')


#     def save_trained_model(self):
#         with open('./models/trained_model.pkl', 'wb') as f:
#             pickle.dump({'model':self.model, 'vectorizer': self.vectorizer}, f)
#         print('Trained model has been saved.')



# train_emotion_model = TrainNBModel()

# train_data_path = './ingest_layer/emotion_train.json'
# train_data = train_emotion_model.get_data(train_data_path)

# trained_model = train_emotion_model.train_model(train_data)

# test_data_path = './ingest_layer/emotion_test.json'
# test_data = train_emotion_model.get_data(test_data_path)

# train_emotion_model.test_model(test_data)

# validation_data_path = './ingest_layer/emotion_validation.json'
# validation_data = train_emotion_model.get_data(validation_data_path)
# train_emotion_model.test_model(validation_data)

# train_emotion_model.save_trained_model()