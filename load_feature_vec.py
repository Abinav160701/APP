import numpy as np
import pickle
def load_features_from_db(l1,l2):
    feature_path=f'Feature_DB/{l1}-{l2}.pkl'
    with open(feature_path, 'rb') as f:
        features, labels = pickle.load(f)
    return features, labels