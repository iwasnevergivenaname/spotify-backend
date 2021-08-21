from flask import Flask, request
import requests
from src.services.user_taste_classifier import GenreClassifier
from src.services.constants import acousticness, danceability, valence, popularity, speechiness, energy

app = Flask(__name__)

model = GenreClassifier()


@app.before_first_request
def model_init():
	model.massage_data()
	model.processing()
	model.train()


frontend_evaluation_endpoint = 'https://spotify-evaluation.herokuapp.com/evaluation'


@app.route('/', methods=["GET"])
def health():
	return "this app works"


@app.route('/predict', methods=["GET", "POST"])
def predict():
	req_data = request.get_json()
	for key in req_data:
		if req_data[key] == None:
			req_data[key] = 0.0
	track = [float(req_data[acousticness]), float(req_data[danceability]), float(req_data[energy]),
	         float(req_data[speechiness]), float(req_data[valence]),
	         float(req_data[popularity])]
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
