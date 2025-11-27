from chatbot import Chatbot
from classifier_layer.news_classifier import NewsClassifier

def chatbot_interface():
    bot = Chatbot()
    
    print('\nWelcome to the News Chatbot. Enter a news article title and some questions you have about it and I will give you some answers!\nTo exit, type exit and press enter')
    
    while True:
        input_string = input('\nWhat would you like to know: ')
        
        if input_string.casefold() != 'exit':
            
            news_title = bot.extract_title(input_string)
            print(news_title)
            
            news_classifier = NewsClassifier('./models/trained_classifier_model.pkl')
            
            topic = news_classifier.classify_news(input_text=news_title)
            
            print(topic)
            
        #     print(f'\nInput: {input_string} \nClassified Topic: {topic.capitalize()}')
        
        else:
            print('\nGoodbye!')
            break

chatbot_interface()