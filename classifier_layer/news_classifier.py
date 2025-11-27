import pickle
import logging

logging.basicConfig(filename='./logging/news_model_logger.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s: %(levelname)s: %(message)s') 

class NewsClassifier:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.vectorizer = data['vectorizer']
        self.logger = logging.getLogger(__name__)


    def classify_news(self, input_text:str):
        
        input_text_vectors = self.vectorizer.transform([input_text])
        
        arr_predicted_topic = self.model.predict(input_text_vectors)
        predicted_topic = arr_predicted_topic[0]

        self.logger.info(f'Input: {input_text}. Result: {predicted_topic}')
        return predicted_topic

    
news=NewsClassifier('./models/trained_classifier_model.pkl')
input_text = 'Champions League review: Arsenal erupt, PSV stun Liverpool and Benfica revive'
topic = news.classify_news(input_text)
print(topic)