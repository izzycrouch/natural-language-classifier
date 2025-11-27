import pickle

class EmotionClassifier:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.vectorizer = data['vectorizer']
        self.label_map = {
            0: 'sadness',
            1: 'joy',
            2: 'love',
            3: 'anger',
            4: 'fear',
            5: 'surprise'
        }


    def classify_emotion(self, input_text:str):
        
        input_text_vectors = self.vectorizer.transform([input_text])
        
        predicted_label_as_array = self.model.predict(input_text_vectors)
        predicted_label = predicted_label_as_array[0]
        emotion = self.label_map[predicted_label]
        
        return emotion
    
    def __str__(self):
        return f'{emotion}'
        
emotions = EmotionClassifier('./trained_model.pkl')
emotion = emotions.classify_emotion(input_text='I am feeling sad.')
print(emotions)