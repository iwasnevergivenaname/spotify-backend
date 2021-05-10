from flask import Flask, request
import requests
from src.services.user_taste_classifier import GenreClassifier

app = Flask(__name__)

model = GenreClassifier()

frontend_evaluation_endpoint = 'http://127.0.0.1:5000/evaluation'


@app.route('/', methods=["GET"])
def health():
	return "this app works"


@app.route('/predict', methods=["GET", "POST"])
def predict():
	model.massage_data()
	model.processing()
	model.train()
	req_data = request.get_json()
	track = [float(req_data["acousticness"]), float(req_data["danceability"]), float(req_data["energy"]),
	         float(req_data["speechiness"]), float(req_data["valence"]),
	         float(req_data["popularity"])]
	prediction = model.predict([track])
	
	resp = requests.post(frontend_evaluation_endpoint, json={'prediction': prediction[0],
	                                                         'track_id': req_data["track_id"],
	                                                         'title': req_data["title"],
	                                                         'user_id': req_data['user_id']
	                                                         })
	if resp.status_code == 200:
		return "done"
	else:
		return "error"
