import pickle
import logging

logging.basicConfig(filename='./logging/RF_model_logger.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s: %(levelname)s: %(message)s') 

class EmotionClassifier:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.vectorizer = data['vectorizer']
        self.label_dict= {0: 'sadness',
                          1: 'joy',
                          2: 'love',
                          3: 'anger',
                          4: 'fear',
                          5: 'surprise'}
        self.logger = logging.getLogger(__name__)
        self.emotion = None


    def classify_emotion(self, input_text:str):
        
        input_text_vectors = self.vectorizer.transform([input_text])
        
        predicted_label_as_array = self.model.predict(input_text_vectors)
        predicted_label = predicted_label_as_array[0]
        emotion = self.label_dict[predicted_label]
        self.logger.info(f'Input:{input_text_vectors}. \n Result: {emotion}')
        self.emotion = emotion
        return self.emotion
    
    def __str__(self):
        return f'{self.emotion}'
    
