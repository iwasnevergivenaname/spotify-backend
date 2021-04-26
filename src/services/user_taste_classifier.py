from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd


# pd.set_option('display.max_rows', genre_data.shape[0]+1)

class GenreClassifier:
	def __init__(self):
		self.path = 'data/data_by_genres.csv'
		self.data = pd.read_csv(self.path)
		self.df_x = None
		self.df_y = None
		self.dt = None
	
	def massage_data(self):
		# lawful good
		self.data['genres'] = np.where(
			((self.data.danceability < 0.4) & (self.data.valence > 0.74) & (self.data.popularity > 74)),
			'popular lawful good', self.data.genres)
		
		self.data['genres'] = np.where(
			((self.data.danceability < 0.4) & (self.data.valence > 0.74) & (self.data.popularity < 75) & (
					self.data.popularity > 29)),
			'avg lawful good', self.data.genres)
		
		self.data['genres'] = np.where(
			((self.data.danceability < 0.4) & (self.data.valence > 0.74) & (self.data.popularity < 30)),
			'deep lawful good', self.data.genres)
		
		# lawful neutral
		self.data['genres'] = np.where(
			((self.data.danceability < 0.4) & (self.data.valence > 0.39) & (self.data.valence < 0.75) & (self.data.popularity > 74)),
			'popular lawful neutral', self.data.genres)
		
		self.data['genres'] = np.where(
			((self.data.danceability < 0.4) & (self.data.valence > 0.39) & (self.data.valence < 0.75) & (
						self.data.popularity < 75) & (self.data.popularity > 29)),
			'avg lawful neutral', self.data.genres)
		
		self.data['genres'] = np.where(
			((self.data.danceability < 0.4) & (self.data.valence > 0.39) & (self.data.valence < 0.75) & (
						self.data.popularity < 30)),
			'deep lawful neutral', self.data.genres)
		
		# lawful evil
		self.data['genres'] = np.where(
			((self.data.danceability < 0.4) & (self.data.valence < 0.4) & (self.data.popularity > 74)),
			'popular lawful evil', self.data.genres)
		
		self.data['genres'] = np.where(
			((self.data.danceability < 0.4) & (self.data.valence < 0.4) & (self.data.popularity < 75) & (self.data.popularity > 29)),
			'avg lawful evil', self.data.genres)
		
		self.data['genres'] = np.where(
			((self.data.danceability < 0.4) & (self.data.valence < 0.4) & (self.data.popularity < 30)),
			'deep lawful evil', self.data.genres)
		
		# neutral good
		self.data['genres'] = np.where(
			((self.data.danceability > 0.39) & (self.data.danceability < 0.75) & (self.data.valence > 0.74) & (self.data.popularity > 74)),
			'popular neutral good', self.data.genres)
		
		self.data['genres'] = np.where(
			((self.data.danceability > 0.39) & (self.data.danceability < 0.75) & (self.data.valence > 0.74) & (self.data.popularity < 75) & (self.data.popularity > 29)),
			'avg neutral good', self.data.genres)
		
		self.data['genres'] = np.where(
			((self.data.danceability > 0.39) & (self.data.danceability < 0.75) & (self.data.valence > 0.74) & (self.data.popularity < 30)),
			'deep neutral good', self.data.genres)
		
		# true neutral
		self.data['genres'] = np.where(
			((self.data.danceability > 0.39) & (self.data.danceability < 0.75) & (self.data.valence > 0.39)
			 & (self.data.valence < 0.75)  & (self.data.popularity > 74)), 'popular true neutral', self.data.genres)
		
		self.data['genres'] = np.where(
			((self.data.danceability > 0.39) & (self.data.danceability < 0.75) & (self.data.valence > 0.39)
			 & (self.data.valence < 0.75) & (self.data.popularity < 75) & (self.data.popularity > 29)), 'avg true neutral', self.data.genres)
		
		self.data['genres'] = np.where(
			((self.data.danceability > 0.39) & (self.data.danceability < 0.75) & (self.data.valence > 0.39)
			 & (self.data.valence < 0.75) & (self.data.popularity < 30)), 'deep true neutral', self.data.genres)
		
		# neutral evil
		self.data['genres'] = np.where(
			((self.data.danceability > 0.39) & (self.data.danceability < 0.75) & (self.data.valence < 0.4) & (self.data.popularity > 74)),
			'popular neutral evil', self.data.genres)
		
		self.data['genres'] = np.where(
			((self.data.danceability > 0.39) & (self.data.danceability < 0.75) & (self.data.valence < 0.4)
			 & (self.data.popularity < 75) & (self.data.popularity > 29)),
			'avg neutral evil', self.data.genres)
		
		self.data['genres'] = np.where(
			((self.data.danceability > 0.39) & (self.data.danceability < 0.75) & (self.data.valence < 0.4) & (self.data.popularity < 30)),
			'deep neutral evil', self.data.genres)
		
		# chaotic good
		self.data['genres'] = np.where(
			((self.data.danceability > 0.74) & (self.data.valence > 0.74) & (self.data.popularity > 74)),
			'popular lawful good', self.data.genres)
		
		self.data['genres'] = np.where(
			((self.data.danceability > 0.74) & (self.data.valence > 0.74) & (self.data.popularity < 75) & (self.data.popularity > 29)),
			'avg lawful good', self.data.genres)
		
		self.data['genres'] = np.where(
			((self.data.danceability > 0.74) & (self.data.valence > 0.74) & (self.data.popularity < 30)),
			'deep lawful good', self.data.genres)
		
		# chaotic neutral
		self.data['genres'] = np.where(
			((self.data.danceability > 0.74) & (self.data.valence > 0.39) & (self.data.valence < 0.75) & (self.data.popularity > 74)),
			'popular lawful neutral', self.data.genres)
		
		self.data['genres'] = np.where(
			((self.data.danceability > 0.74) & (self.data.valence > 0.39) & (self.data.valence < 0.75) & (self.data.popularity < 75) & (self.data.popularity > 29)),
			'avg lawful neutral', self.data.genres)
		
		self.data['genres'] = np.where(
			((self.data.danceability > 0.74) & (self.data.valence > 0.39) & (self.data.valence < 0.75) & (self.data.popularity < 30)),
			'deep lawful neutral', self.data.genres)
		
		# chaotic evil
		self.data['genres'] = np.where(
			((self.data.danceability > 0.74) & (self.data.valence < 0.4) & (self.data.popularity > 74)),
			'popular lawful good', self.data.genres)
		
		self.data['genres'] = np.where(
			((self.data.danceability > 0.74) & (self.data.valence < 0.4) & (self.data.popularity < 75) & (self.data.popularity > 29)),
			'avg lawful good', self.data.genres)
		
		self.data['genres'] = np.where(
			((self.data.danceability > 0.74) & (self.data.valence < 0.4) & (self.data.popularity < 30)),
			'deep lawful good', self.data.genres)
		
	
	def processing(self):
		cols = ["acousticness", "danceability", "energy", "speechiness", "valence", "popularity"]
		self.df_x = self.data[cols].values
		self.df_y = self.data[['genres']].values
	
	def train(self):
		self.dt = DecisionTreeClassifier()
		self.dt.fit(self.df_x[:1000], self.df_y[:1000])
	
	def predict(self, track):
		return self.dt.predict(track)
