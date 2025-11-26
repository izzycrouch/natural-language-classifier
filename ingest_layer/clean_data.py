import json

def clean_data(input_file_path, output_file_path):
    with open(input_file_path, mode='r', encoding='utf-8') as f:
        list_data = json.load(f)
        cleaned_data = []
        for data in list_data:
            if isinstance(data['text'], str) and isinstance(data['label'], int):
                cleaned_data.append(data)
    
    with open(output_file_path, mode='w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=4)

clean_data('./ingest_layer/load_emotion_train.json', './ingest_layer/emotion_train.json')
clean_data('./ingest_layer/load_emotion_test.json', './ingest_layer/emotion_test.json')
clean_data('./ingest_layer/load_emotion_validation.json', './ingest_layer/emotion_validation.json')