from datasets import load_dataset
import json

def load_data(dataset, output_path_prefix):   
    dict_keys =  list(dataset.keys())
    
    for key in dict_keys:
        list_items = ds[key].to_list()
        with open(f'{output_path_prefix}{key}.json', mode='w', encoding='utf-8') as f:
            json.dump(list_items, f, indent=4)


ds = load_dataset("dair-ai/emotion", "split")

load_data(ds, './ingest_layer/load_emotion_')

def clean_data(input_file_path):
    with open(input_file_path, mode='r', encoding='utf-8') as f:
        list_data = json.load(f)
        cleaned_data = []
        for data in list_data:
            if isinstance(data['text'], str) and isinstance(data['label'], int):
                cleaned_data.append(data)
    return cleaned_data


def save_data(input_file_path, output_file_path):
    cleaned_data = clean_data(input_file_path)
    with open(output_file_path, mode='w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=4)

save_data('./ingest_layer/load_emotion_train.json', './ingest_layer/emotion_train.json')
save_data('./ingest_layer/load_emotion_test.json', './ingest_layer/emotion_test.json')
save_data('./ingest_layer/load_emotion_validation.json', './ingest_layer/emotion_validation.json')