from chatbot import Chatbot
from classifier_layer.news_classifier import NewsClassifier

def chatbot_interface():
    bot = Chatbot()
    
    print('\n-----------------------------------------')
    print('\n             NEWS CHATBOT')
    print('\n-----------------------------------------')
    
    print('\nWelcome to News Chatbot! Enter a news headline and some questions you have about it and I will give you some answers!\nPlease enclose the news headline/article title in quotation marks.')
    print('To exit, type exit when prompted for input and press enter.')
    
    print(bot.chat_history_ids)

    while True:
        input_string = input('\nBot: What would you like to know? ')
        
        if input_string.casefold() != 'exit':
            
            news_title = bot.generate_reply(is_extract=True, prompt=input_string)

            while news_title == 'None':
                print('Bot: I\'m sorry, I couldnt detect a news headline in your question. Try again.')
                
                input_string = input('\nBot: What would you like to know? ')
                
                if input_string.casefold() != 'exit':

                    news_title = bot.generate_reply(is_extract=True, prompt=input_string)
                
                else:
                    return
            
            news_classifier = NewsClassifier('./models/trained_classifier_model.pkl')
            
            topic = news_classifier.classify_news(input_text=news_title)
            
            print(f'\nTitle: {news_title} \nClassified Topic: {topic.capitalize()}\n')

            reply = bot.generate_reply(is_extract=False, prompt=input_string)
            print(f'\nBot: {reply}')

            while True:
                another_question = input('\nBot: Do you have anymore questions on this article? (y/n) ')
                
                if another_question.casefold() == 'exit':
                    return
                
                elif another_question.casefold() == 'y':
                    input_string = input('\nBot: What else would you like to know about this article? ')
                    reply = bot.generate_reply(is_extract=False, prompt=input_string)
                    print(f'Bot: {reply}')
                
                else:
                    break 
                

        else:
            print('\nGoodbye!')
            break

chatbot_interface()

