import joblib

model = joblib.load("alphabet.pkl")

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def predict_alphabet(img):
    pred = model.predict(img)[0]

    if isinstance(pred, str):
        return pred

    return labels[int(pred)]