import pickle

with open('./trained_model.pkl', 'rb') as f:
    model = pickle.load(f)