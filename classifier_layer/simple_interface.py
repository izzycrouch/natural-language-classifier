from emotion_classifier import EmotionClassifier

def interface():
    print('\nWelcome to Emotion Classifier! \n\nEnter some text and I will evaluate what emotion it classifies under. To exit, type exit and press enter.')
    while True:
        input_string = input('\nEnter a sentence to classify: ')
        
        if input_string.casefold() != 'exit':
            
            emotion_classifier = EmotionClassifier('./models/trained_classifier_model.pkl')
            emotion = emotion_classifier.classify_emotion(input_text=input_string)
            print(f'\nInput: {input_string} \nClassified Emotion: {emotion.capitalize()}')
        
        else:
            print('\nGoodbye!')
            break

interface()