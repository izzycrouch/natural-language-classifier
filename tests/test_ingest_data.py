from ingest_layer.ingest_data import clean_data
import json

def test_clean_data_function_returns_a_list_of_dictionaries():
    input_path = './tests/test_data.json'
    cleaned_data = clean_data(input_path)
    assert isinstance(cleaned_data, list)


def test_clean_data_function_returns_a_list_of_dictionaries_with_correct_keys():
    input_path = './tests/test_data.json'
    cleaned_data = clean_data(input_path)
    list_keys = []
    for d in cleaned_data:
        keys = list(d.keys())
        for k in keys:
            if k not in list_keys:
                list_keys.append(k)
    assert list_keys == ["text", "label"]


def test_clean_data_function_returns_correct_data_types_for_dictionary_values():
    input_path = './tests/test_data.json'
    cleaned_data = clean_data(input_path)
    
    correct_data_types = False
    for i in cleaned_data:
        if isinstance(i["text"], str) and isinstance(i["label"], int):
            correct_data_types = True
    
    assert correct_data_types == True


def test_clean_data_function_returns_all_data_if_data_is_valid():
    input_path = './tests/test_data.json'
    cleaned_data = clean_data(input_path)
    
    with open (input_path, mode='r') as f:
        list_data = json.load(f)

    assert len(list_data) == len(cleaned_data)


def test_clean_data_function_removes_label_values_if_not_int():
    input_path = './tests/test_data_not_clean.json'
    cleaned_data = clean_data(input_path)

    with open (input_path, mode='r') as f:
        list_data = json.load(f)
    
    list_label_vals =  [d["label"] for d in cleaned_data]
    correct_data_types = False
    for v in list_label_vals:
        if isinstance(v, int):
            correct_data_types = True

    assert len(list_data) != len(cleaned_data)
    assert correct_data_types == True


def test_clean_data_function_removes_text_values_if_not_valid_string():
    input_path = './tests/test_data_not_clean.json'
    cleaned_data = clean_data(input_path)

    with open (input_path, mode='r') as f:
        list_data = json.load(f)
    
    list_text_vals =  [d["text"] for d in cleaned_data]
    print(list_text_vals)
    correct_data_types = False
    for v in list_text_vals:
        if isinstance(v, str) and v != "":
            correct_data_types = True

    assert len(list_data) != len(cleaned_data)
    assert correct_data_types == True


def test_clean_data_function_removes_data_with_invalid_keys():
    input_path = './tests/test_data_not_clean.json'
    cleaned_data = clean_data(input_path)
   
    list_keys = []
    for d in cleaned_data:
        keys = list(d.keys())
        for k in keys:
            if k not in list_keys:
                list_keys.append(k)
    assert list_keys == ["text", "label"]