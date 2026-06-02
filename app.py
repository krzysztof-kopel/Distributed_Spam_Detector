import os

os.environ["TF_USE_LEGACY_KERAS"] = "1" # Only legacy Keras works well with Ray

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
import ray
import tensorflow as tf

class EmailService(BaseModel):
    text: str

app = FastAPI()

@ray.serve.deployment
class SpamClassifierService:
    def __init__(self):
        self.model = tf.keras.models.load_model(os.path.abspath("artifacts/model.keras"))
        self.vectorizer = tf.keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=200)
        with open(os.path.abspath("artifacts/vocabulary.txt"), encoding="utf-8") as file:
            raw_vocabulary = file.read().splitlines()
        vocabulary = [
            token.strip() for token in raw_vocabulary
            if token.strip() and token.strip() not in ("", "[UNK]")
        ]
        self.vectorizer.set_vocabulary(vocabulary)

    def predict_spam(self, payload: EmailService):
        raw_text = payload.text
        text_vectorized = self.vectorizer([raw_text]).numpy()
        result = self.model.predict(text_vectorized)[0][0]
        result = float(result)
        return {"Spam probability": result, "result": "Spam" if result > 0.5 else "Normal email"}


@ray.serve.deployment
@ray.serve.ingress(app)
class APIIngres:
    def __init__(self, model_handle):
        self.model_handle = model_handle

    @app.post("/predict")
    async def predict(self, payload: EmailService):
        return await self.model_handle.predict_spam.remote(payload)

ray.serve.start(http_options={"host": "0.0.0.0", "port": 8000})
model_deployment = SpamClassifierService.bind()
deployment_instance = APIIngres.bind(model_deployment)
