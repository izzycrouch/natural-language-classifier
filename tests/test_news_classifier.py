from classifier_layer.news_classifier import NewsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

def test_news_classifier_model_is_an_isinstance_of_a_naive_bayes_model():
    test_nc = NewsClassifier('./models/trained_classifier_model.pkl')
    assert isinstance(test_nc.model, MultinomialNB)

def test_news_classifier_vectorizer_is_an_isinstance_of_a_CountVectorizer():
    test_nc = NewsClassifier('./models/trained_classifier_model.pkl')
    assert isinstance(test_nc.vectorizer, CountVectorizer)

def test_news_classifier_classifies_input_string_correctly_1():
    test_nc = NewsClassifier('./models/trained_classifier_model.pkl')
    topic = test_nc.classify_news(input_text='Champions League review: Arsenal erupt, PSV stun Liverpool and Benfica revive')
    assert topic == 'SPORTS'

def test_news_classifier_classifies_input_string_correctly_2():
    test_nc = NewsClassifier('./models/trained_classifier_model.pkl')
    topic = test_nc.classify_news(input_text='The apple smart phone to get this year')
    assert topic == 'TECHNOLOGY'

def test_news_classifier_classifies_input_string_correctly_3():
    test_nc = NewsClassifier('./models/trained_classifier_model.pkl')
    topic = test_nc.classify_news(input_text='What is prostate cancer and how is it diagnosed in the UK?')
    assert topic == 'HEALTH'

def test_news_classifier_classifies_input_string_correctly_4():
    test_nc = NewsClassifier('./models/trained_classifier_model.pkl')
    topic = test_nc.classify_news(input_text='Income tax threshold freeze will hit poorer households harder, experts say')
    assert topic == 'NATION'

def test_news_classifier_classifies_input_string_correctly_5():
    test_nc = NewsClassifier('./models/trained_classifier_model.pkl')
    topic = test_nc.classify_news(input_text='You\'re gonna need a bigger boat: the 20 best films set on water - ranked!')
    assert topic == 'ENTERTAINMENT'