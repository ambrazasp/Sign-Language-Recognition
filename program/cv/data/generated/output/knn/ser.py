import pickle
import sklearn

with open('model-serialized-knn.pkl', 'rb') as f:
    data = pickle.load(f)
