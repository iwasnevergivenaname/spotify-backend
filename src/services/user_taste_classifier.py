from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


# pd.set_option('display.max_rows', genre_data.shape[0]+1)

class GenreClassifier:
	def __init__(self):
		self.path = 'data/data_by_genres.csv'
		self.data = pd.read_csv(self.path)
		# self.data = None
		self.df_x = None
		self.df_y = None
		self.dt = None
	
	def massage_data(self):
		# lawful good
		self.data['genres'] = np.where(
			((self.data.danceability < 0.4) & (self.data.valence < 0.4) & (self.data.popularity > 70)), 'lawful good',
			self.data.genres)
		# lawful neutral
		self.data['genres'] = np.where(
			((self.data.danceability < 0.4) & (self.data.valence >= 0.4) & (self.data.valence <= 0.7)), 'lawful neutral',
			self.data.genres)
		#  lawful evil
		self.data['genres'] = np.where(
			((self.data.danceability < 0.4) & (self.data.valence > 0.7) & (self.data.energy < .3)), 'lawful evil',
			self.data.genres)
		# neutral good
		self.data['genres'] = np.where(
			((self.data.danceability >= 0.4) & (self.data.danceability <= 0.7) & (self.data.valence < 0.4)), 'neutral good',
			self.data.genres)
		# true neutral
		self.data['genres'] = np.where(
			((self.data.danceability >= 0.4) & (self.data.danceability <= 0.7) & (self.data.valence >= 0.4) & (
						self.data.valence <= 0.7)), 'true neutral', self.data.genres)
		# neutral evil
		self.data['genres'] = np.where(
			((self.data.danceability >= 0.4) & (self.data.danceability <= 0.7) & (self.data.valence > 0.7) & (
						self.data.energy < .3)), 'neutral evil', self.data.genres)
		# chaotic good
		self.data['genres'] = np.where(
			((self.data.danceability > 0.7) & (self.data.valence < 0.4) & (self.data.popularity < 30)), 'chaotic good',
			self.data.genres)
		# chaotic neutral
		self.data['genres'] = np.where(
			((self.data.danceability > 0.7) & (self.data.valence >= 0.4) & (self.data.valence <= 0.7) & (
						self.data.popularity < 30)), 'chaotic neutral', self.data.genres)
		# chaotic evil
		self.data['genres'] = np.where(((self.data.danceability > 0.7) & (self.data.valence > 0.7) & (
					self.data.popularity < 30) & (self.data.energy < .3)), 'chaotic evil', self.data.genres)

	
	def processing(self):
		cols = ["acousticness", "danceability", "energy", "speechiness", "valence", "popularity"]
		self.df_x = self.data[cols].values
		self.df_y = self.data[['genres']].values
	
	def train(self):
		self.dt = DecisionTreeClassifier()
		self.dt.fit(self.df_x[:1000], self.df_y[:1000])
	
	def predict(self, track):
		return self.dt.predict(track)
