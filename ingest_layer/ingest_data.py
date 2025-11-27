import json
import csv
from datetime import datetime

def convert_CSV_to_JSON(input_path, output_path):   
    data_dict = []
    with open(input_path, mode='r', newline='', encoding='utf-8') as f:
      dict_reader = csv.DictReader(f, delimiter=';')
      for rows in dict_reader:
          data_dict.append(rows)
    
    with open(output_path, mode='w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=4)
    
    print('CSV file converted to JSON file.')    


# input_path = './ingest_layer/labeled_newscatcher_dataset.csv'
# output_path = './ingest_layer/labeled_newscatcher_dataset.json'
# convert_CSV_to_JSON(input_path, output_path)


def clean_data(input_path):
    with open(input_path, mode='r', encoding='utf-8') as f:
        list_data = json.load(f)

    valid_keys = ['topic', 'link', 'domain', 'published_date', 'title', 'lang']
    valid_topics = ['BUSINESS', 'ENTERTAINMENT', 'HEALTH', 'NATION', 'SCIENCE', 'SPORTS', 'TECHNOLOGY']
    
    cleaned_data = []
    
    for d in list_data:
        
        dict_keys = list(d.keys())
        
        if dict_keys == valid_keys:
            
            if d["topic"] in valid_topics:
                cleaned_data.append(d)

    for d in cleaned_data:
        for key in d:
            if d[key] == '':
                cleaned_data.remove(d)

    for d in cleaned_data:
        published_date = datetime.strptime(d['published_date'], '%Y-%m-%d %H:%M:%S')
        d['published_date'] = published_date
    
    return cleaned_data

def extract_relevant_data(cleaned_data):
    extracted_data = []
    for d in cleaned_data:
        relevant_keys = ['topic', 'title']
        new_d  = {key: d[key] for key in relevant_keys}
        extracted_data.append(new_d)
    return extracted_data


def save_data(input_file_path, output_file_path):
    print('Start cleaning data...')
    cleaned_data = clean_data(input_file_path)
    print('Data has been cleaned.')
    print('Start extracting data...')
    extracted_data = extract_relevant_data(cleaned_data)
    print('Data has been extracted.')
    with open(output_file_path, mode='w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=4)

save_data('./ingest_layer/labeled_newscatcher_dataset.json', './ingest_layer/cleaned_labeled_newscatcher_dataset.json')
