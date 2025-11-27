from ingest_layer.ingest_data import clean_data, extract_relevant_data
import json
from datetime import datetime

def test_clean_data_function_returns_a_list_of_dictionaries():
    input_path = './tests/test_data.json'
    cleaned_data = clean_data(input_path)
    assert isinstance(cleaned_data, list)
    assert isinstance(cleaned_data[0], dict)


def test_clean_data_function_returns_a_list_of_dictionaries_with_correct_keys():
    input_path = './tests/test_data.json'
    cleaned_data = clean_data(input_path)
    list_keys = []
    for d in cleaned_data:
        keys = list(d.keys())
        for k in keys:
            if k not in list_keys:
                list_keys.append(k)
    assert list_keys == ['topic', 'link', 'domain', 'published_date', 'title', 'lang']

def test_clean_data_function_removes_dictionaries_with_incorrect_key_names():
    input_path = './tests/test_data_not_clean.json'
    cleaned_data = clean_data(input_path)
    list_keys = []
    for d in cleaned_data:
        keys = list(d.keys())
        for k in keys:
            if k not in list_keys:
                list_keys.append(k)
    
    assert list_keys == ['topic', 'link', 'domain', 'published_date', 'title', 'lang']
    
    with open (input_path, mode='r') as f:
        list_data = json.load(f)
    
    assert len(cleaned_data) != len(list_data)

def test_clean_data_function_removes_dictionaries_with_invalid_topic_value():
    input_path = './tests/test_data_not_clean.json'
    cleaned_data = clean_data(input_path)
    
    topics = set(data['topic'] for data in cleaned_data)
    
    valid_topics = ['BUSINESS', 'ENTERTAINMENT', 'HEALTH', 'NATION', 'SCIENCE', 'SPORTS', 'TECHNOLOGY']
    
    only_valid_topics = True
    
    for topic in topics:
        if topic not in valid_topics:
            only_valid_topics = False

    assert only_valid_topics == True

    with open (input_path, mode='r') as f:
        list_data = json.load(f)
    
    assert len(cleaned_data) != len(list_data)

def test_clean_data_function_removed_data_which_contains_empty_str():
    input_path = './tests/test_data_not_clean.json'
    cleaned_data = clean_data(input_path)
    
    list_values = []
    for d in cleaned_data:
        values = list(d.values())
        for v in values:
            list_values.append(v)

    assert '' not in list_values


def test_clean_data_function_returns_correct_data_type_for_published_date():
    input_path = './tests/test_data.json'
    cleaned_data = clean_data(input_path)
    
    correct_data_types = False
    for d in cleaned_data:
        if isinstance(d["published_date"], datetime):
            correct_data_types = True
    
    assert correct_data_types == True


def test_clean_data_function_returns_all_data_if_data_is_valid():
    input_path = './tests/test_data.json'
    cleaned_data = clean_data(input_path)
    
    with open (input_path, mode='r') as f:
        list_data = json.load(f)

    assert len(list_data) == len(cleaned_data)


def test_extract_relevant_data_returns_list_of_dictionaries():
    input_path = './tests/test_data.json'
    cleaned_data = clean_data(input_path)
    extracted_data = extract_relevant_data(cleaned_data)
    assert isinstance(extracted_data, list)
    assert isinstance(extracted_data[0], dict)


def test_extract_relevant_data_returns_same_num_dicts_as_clean_data():
    input_path = './tests/test_data.json'
    cleaned_data = clean_data(input_path)
    extracted_data = extract_relevant_data(cleaned_data)
    assert len(extracted_data) == len(cleaned_data)


def test_extract_relevant_data_returns_only_relevant_dict_keys():
    input_path = './tests/test_data.json'
    cleaned_data = clean_data(input_path)
    extracted_data = extract_relevant_data(cleaned_data)
    list_keys = []
    for d in extracted_data:
        for key in d:
            if key not in list_keys:
                list_keys.append(key)
    
    assert list_keys == ['topic', 'title']