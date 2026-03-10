from fastapi import FastAPI
import numpy as np
import tensorflow as tf

app = FastAPI()

# modelo simple cargado
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation="sigmoid", input_shape=(3,)),
    tf.keras.layers.Dense(3, activation="sigmoid")
])

@app.get("/")
def home():
    return {"message": "TensorFlow API running"}

@app.post("/predict")
def predict(data: dict):

    x = np.array([data["input"]])
    y = model.predict(x)

    return {
        "input": data["input"],
        "output": y.tolist()
    }