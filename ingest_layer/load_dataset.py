from datasets import load_dataset
import json


def load_data():   
    ds = load_dataset("dair-ai/emotion", "split")
    
    # train data
    train_list = ds['train'].to_list() 
    with open('./ingest_layer/load_emotion_train.json', mode='w', encoding='utf-8') as f:
        json.dump(train_list, f, indent=4)
    
    # validation data
    validation_list = ds['validation'].to_list() 
    with open('./ingest_layer/load_emotion_validation.json', mode='w', encoding='utf-8') as f:
        json.dump(validation_list, f, indent=4)

    # test data
    test_list = ds['test'].to_list() 
    with open('./ingest_layer/load_emotion_test.json', mode='w', encoding='utf-8') as f:
        json.dump(test_list, f, indent=4)

load_data()