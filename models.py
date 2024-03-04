from tensorflow.keras.models import load_model
def load_model_from_db(l1,l2):
    model_path=f'Models_DB/{l1}-{l2}.h5'
    return load_model(model_path)