from sklearn.feature_extraction.text import CountVectorizer
import pickle

class EmotionClassifier:
    def __init__(self):
        with open('./trained_model.pkl', 'rb') as f:
            ss = pickle.load(f)
            print(ss['model'])
        
    #     self.label_map = {
    #         0: 'sadness',
    #         1: 'joy',
    #         2: 'love',
    #         3: 'anger',
    #         4: 'fear',
    #         5: 'surprise'
    #     }

    #     self.vectorizer = CountVectorizer()

    # def classify_emotion(self, input_text:str):
        
    #     input_text_vectors = self.vectorizer.fit_transform([input_text])
        
    #     predicted_label = self.model.predict(input_text_vectors)
    #     emotion = self.label_map.get(predicted_label, )
        # return emotion
        
emotions = EmotionClassifier()
# emotion = emotions.classify_emotion(input_text='I am feeling sad.')
print(emotions)