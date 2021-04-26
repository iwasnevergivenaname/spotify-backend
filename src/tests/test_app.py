"""python3 -m unittest src/tests/test_app.py """

from src.app import app
import json
from unittest import TestCase

app.config['TESTING'] = True

class TestPredictionTestCase(TestCase):
	"""testing how the model handles and returns predictions"""
	
	def test_model_received_data(self):
		with app.test_client() as client:
			payload = json.dumps({'title': 'Backseat (feat. Carly Rae Jepsen)', 'acousticness': '0.247',
			                      'danceability': '0.626', 'energy': '0.607', 'speechiness': '0.0409', 'valence': '0.339',
			                      'popularity': '49', 'track_id': '4HjtHraeKy5wA4DA9o92HZ', 'user_id': 'user1'})
			req = client.post('/predict', json=payload)
			# html = resp.get_data(as_text=True)
			
			self.assertEqual(req.status_code, 200)