from flask import Flask, redirect, request, json
import requests
from services.user_taste_classifier import GenreClassifier

app = Flask(__name__)

model = GenreClassifier()

frontend_evaluation_endpoint = 'http://127.0.0.1:5000/evaluation'

@app.route('/')
def home():
	model.massage_data()
	model.processing()
	model.train()
	return redirect("/train")


@app.route('/predict', methods=["GET", "POST"])
def predict():
	model.massage_data()
	model.processing()
	model.train()
	req_data = request.get_json()
	track = [float(req_data["acousticness"]), float(req_data["danceability"]), float(req_data["energy"]),
	         float(req_data["speechiness"]), float(req_data["valence"]),
	         float(req_data["popularity"])]
	print("🌸", req_data["title"], track, "🎀")
	print("💕", model.predict([track]), "✨")
	prediction = model.predict([track])
	resp = requests.post(frontend_evaluation_endpoint, data=prediction[0])
	print("🍄", resp)
	return "<h1>time to train the model</h1>"


if __name__ == '__main__':
	app.run(port=8000)
