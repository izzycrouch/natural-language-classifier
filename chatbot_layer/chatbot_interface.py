from chatbot import Chatbot
from classifier_layer.news_classifier import NewsClassifier

def chatbot_interface():
    bot = Chatbot()
    
    print('\nWelcome to the News Chatbot. Enter a news article title and some questions you have about it and I will give you some answers!\nTo exit, type exit and press enter')
    
    while True:
        input_string = input('\nWhat would you like to know: ')
        
        if input_string.casefold() != 'exit':
            
            news_title = bot.extract_title(input_string)
            
            news_classifier = NewsClassifier('./models/trained_classifier_model.pkl')
            
            topic = news_classifier.classify_news(input_text=news_title)
            
            print(f'\nTitle: {news_title} \nClassified Topic: {topic.capitalize()}')

            # reply = bot.generate_reply(input_string)
            # print(f'Response: {reply}')
        
        else:
            print('\nGoodbye!')
            break

chatbot_interface()