from news_classifier import NewsClassifier

def interface():
    print('\nWelcome to News Classifier! \n\Input a title of a news article (then press enter) and I will evaluate what topic the article classifies under. To exit, type exit and press enter.')
    while True:
        input_string = input('\nEnter a title to classify: ')
        
        if input_string.casefold() != 'exit':
            
            news_classifier = NewsClassifier('./models/trained_classifier_model.pkl')
            topic = news_classifier.classify_news(input_text=input_string)
            print(f'\nInput: {input_string} \nClassified Topic: {topic.capitalize()}')
        
        else:
            print('\nGoodbye!')
            break

interface()