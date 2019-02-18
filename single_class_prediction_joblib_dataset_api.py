from sklearn.externals import joblib


def get_prediction(xnew):
    # load the model from disk
    loaded_model = joblib.load('joblib_iris_flowers_finalized_model.sav')
    # make a prediction
    return loaded_model.predict(xnew)[0]
