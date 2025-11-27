from classifier_layer.emotion_classifier import EmotionClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

def test_emotion_classifier_model_is_an_isinstance_of_a_random_forest_model():
    test_e = EmotionClassifier('./models/trained_classifier_model.pkl')
    assert isinstance(test_e.model, RandomForestClassifier)

def test_emotion_classifier_vectorizer_is_an_isinstance_of_a_CountVectorizer():
    test_e = EmotionClassifier('./models/trained_classifier_model.pkl')
    assert isinstance(test_e.vectorizer, CountVectorizer)

def test_emotion_classifier_classifies_input_string_correctly():
    test_e = EmotionClassifier('./models/trained_classifier_model.pkl')
    emotion = test_e.classify_emotion(input_text='I am feeling sad.')
    assert emotion == 'sadness'

def test_emotion_classifier_classifies_input_string_correctly_2():
    test_e = EmotionClassifier('./models/trained_classifier_model.pkl')
    emotion = test_e.classify_emotion(input_text='I am afraid of spiders.')
    assert emotion == 'fear'

def test_emotion_classifier_classifies_input_string_correctly_4():
    test_e = EmotionClassifier('./models/trained_classifier_model.pkl')
    emotion = test_e.classify_emotion(input_text='dogs make me laugh.')
    assert emotion == 'joy'