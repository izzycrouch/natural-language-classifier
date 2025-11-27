from news_classifier import NewsClassifier

def interface():
    print('\nWelcome to Emotion Classifier! \n\Input some text (then press enter) and I will evaluate what emotion it classifies under. To exit, type exit and press enter.')
    while True:
        input_string = input('\nEnter a sentence to classify: ')
        
        if input_string.casefold() != 'exit':
            
            news_classifier = NewsClassifier('./models/trained_classifier_model.pkl')
            topic = news_classifier.classify_news(input_text=input_string)
            print(f'\nInput: {input_string} \nClassified Topic: {topic.capitalize()}')
        
        else:
            print('\nGoodbye!')
            break

interface()