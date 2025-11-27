from classifier_layer.emotion_classifier import EmotionClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

def test_emotion_classifier_model_is_an_isinstance_of_a_naive_bayes_model():
    test_e = EmotionClassifier('./trained_model.pkl')
    assert isinstance(test_e.model, MultinomialNB)

def test_emotion_classifier_vectorizer_is_an_isinstance_of_a_CountVectorizer():
    test_e = EmotionClassifier('./trained_model.pkl')
    assert isinstance(test_e.vectorizer, CountVectorizer)

def test_emotion_classifier_classifies_input_string_correctly():
    test_e = EmotionClassifier('./trained_model.pkl')
    emotion = test_e.classify_emotion(input_text='I am feeling sad.')
    assert emotion == 'sadness'

def test_emotion_classifier_classifies_input_string_correctly_2():
    test_e = EmotionClassifier('./trained_model.pkl')
    emotion = test_e.classify_emotion(input_text='I am afraid of spiders.')
    assert emotion == 'fear'

def test_emotion_classifier_classifies_input_string_correctly_3():
    test_e = EmotionClassifier('./trained_model.pkl')
    emotion = test_e.classify_emotion(input_text='I am wondering what the point of life is.')
    assert emotion == 'sadness'

def test_emotion_classifier_classifies_input_string_correctly_4():
    test_e = EmotionClassifier('./trained_model.pkl')
    emotion = test_e.classify_emotion(input_text='dogs make me laugh.')
    assert emotion == 'joy'