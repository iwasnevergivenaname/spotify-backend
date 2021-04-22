from sklearn.tree import DecisionTreeClassifier
import pandas as pd


# pd.set_option('display.max_rows', genre_data.shape[0]+1)

class GenreClassifier:
	def __init__(self):
		self.path = 'data/data_by_genres.csv'
		self.pd = pd.read_csv(self.path)
		self.data = None
		self.df_x = None
		self.df_y = None
		self.dt = None
	
	def massage_data(self):
		self.data = self.pd.replace(
			{'genres': {
				# r'.* deep .*': 'chaotic neutral', r'^deep.*': 'chaotic neutral', r'^deep .*': 'chaotic neutral', r'.* deep': 'chaotic neutral',
				# r'.* classical .*': 'lawful good', r'^classical.*': 'lawful good', r'^classical .*': 'lawful good',
				# r'.* classical': 'lawful good',
				# r'.* punk .*': 'punk', r'^punk.*': 'punk', r'^punk .*': 'punk', r'.* punk': 'punk',
				# r'.* pop .*': 'lawful good', r'^pop.*': 'lawful good', r'^pop .*': 'lawful good', r'.* pop': 'lawful good',
				# r'.* rock .*': 'rock', r'^rock.*': 'rock', r'^rock .*': 'rock', r'.* rock': 'rock',
				# r'.* country .*': 'country', r'^country.*': 'country', r'^country .*': 'country',
				# r'.* country': 'country',
				# r'.* metal .*': 'metal', r'^metal.*': 'metal', r'^metal .*': 'metal', r'.* metal': 'metal',
				# r'.* anime .*': 'chaotic evil', r'^anime.*': 'chaotic evil', r'^anime .*': 'chaotic evil', r'.* anime': 'chaotic evil',
				# r'.* comedy .*': 'comedy', r'^comedy.*': 'comedy', r'^comedy .*': 'comedy', r'.* comedy': 'comedy',
				# r'.* hip hop .*': 'hip hop', r'^hip hop.*': 'hip hop', r'^hip hop .*': 'hip hop',
				# r'.* hip hop': 'hip hop',
				# r'.* folk .*': 'folk', r'^folk.*': 'folk', r'^folk .*': 'folk', r'.* folk': 'folk',
				# r'.* indie .*': 'indie', r'^indie.*': 'indie', r'^indie .*': 'indie', r'.* indie': 'indie',
				# r'.* hardcore .*': 'hardcore', r'^hardcore.*': 'hardcore', r'^hardcore .*': 'hardcore',
				# r'.* hardcore': 'hardcore',
				# r'.* jazz .*': 'jazz', r'^jazz.*': 'jazz', r'^jazz .*': 'jazz', r'.* jazz': 'jazz',
				# r'.* blues .*': 'blues', r'^blues.*': 'blues', r'^blues .*': 'blues', r'.* blues': 'blues',
				# r'.* trap .*': 'trap', r'^trap.*': 'trap', r'^trap .*': 'trap', r'.* trap': 'trap',
				# r'.* house .*': 'house', r'^house.*': 'house', r'^house .*': 'house', r'.* house': 'house',
				# r'.* gospel .*': 'lawful evil', r'^gospel.*': 'lawful evil', r'^gospel .*': 'lawful evil', r'.* gospel': 'lawful evil',
				# r'.* worship .*': 'lawful evil', r'^worship.*': 'lawful evil', r'^worship .*': 'lawful evil',
				# r'.* worship': 'lawful evil',
				# r'.* disco .*': 'disco', r'^disco.*': 'disco', r'^disco .*': 'disco', r'.* disco': 'disco',
				# r'.* techno .*': 'techno', r'^techno.*': 'techno', r'^techno .*': 'techno', r'.* techno': 'techno',
				# r'.* soul .*': 'soul', r'^soul.*': 'soul', r'^soul .*': 'soul', r'.* soul': 'soul',
				# r'.* choir .*': 'choir', r'^choir.*': 'choir', r'^choir .*': 'choir', r'.* choir': 'choir',
				# r'.* ambient .*': 'chaotic neutral', r'^ambient.*': 'chaotic neutral', r'^ambient .*': 'chaotic neutral',
				# r'.* ambient': 'chaotic neutral',
				# r'.* emo .*': 'chaotic good', r'^emo.*': 'chaotic good', r'^emo .*': 'chaotic good', r'.* emo': 'chaotic good',
				# r'.* poetry .*': 'neutral evil', r'^poetry.*': 'neutral evil', r'^poetry .*': 'neutral evil', r'.* poetry': 'neutral evil',
				# r'.* electronic .*': 'chaotic good', r'^electronic.*': 'chaotic good', r'^electronic .*': 'chaotic good',
				# r'.* electronic': 'chaotic good',
				# r'.* rap .*': 'rap', r'^rap.*': 'rap', r'^rap .*': 'rap', r'.* rap': 'rap',
				# r'.* ska .*': 'chaotic evil', r'^ska.*': 'chaotic evil', r'^ska .*': 'chaotic evil', r'.* ska': 'chaotic evil',
				# r'.* afro .*': 'afro', r'^afro.*': 'afro', r'^afro .*': 'afro', r'.* afro': 'afro',
				# r'.* r&b .*': 'true neutral', r'^r&b.*': 'true neutral', r'^r&b .*': 'true neutral', r'.* r&b': 'true neutral',
				r'^0.*': 'lawful good', r'^9.*': 'lawful good', r'^i.*': 'lawful good', r'^q.*': 'lawful good',
				r'^s.*': 'lawful evil', r'^1.*': 'lawful neutral', r'^a.*': 'lawful neutral', r'^j.*': 'lawful neutral',
				r'^r.*': 'lawful neutral', r'^2.*': 'lawful evil', r'^b.*': 'lawful evil', r'^k.*': 'lawful evil',
				r'^3.*': 'neutral good', r'^c.*': 'neutral good', r'^l.*': 'neutral good', r'^t.*': 'neutral good',
				r'^%.*': 'neutral good', r'^4.*': ' true neutral', r'^d.*': ' true neutral', r'^m.*': ' true neutral',
				r'^&.*': ' true neutral', r'^u.*': ' true neutral', r'^5.*': ' neutral evil', r'^e.*': ' neutral evil',
				r'^n.*': ' neutral evil', r'^v.*': ' neutral evil', r'^6.*': ' chaotic good', r'^f.*': ' chaotic good',
				r'^o.*': ' chaotic good', r'^w.*': ' chaotic good', r'^7.*': ' chaotic neutral', r'^g.*': ' chaotic neutral',
				r'^z.*': ' chaotic neutral', r'^x.*': ' chaotic neutral', r'^8.*': ' chaotic evil', r'^h.*': ' chaotic evil',
				r'^p.*': ' chaotic evil', r'^y.*': ' chaotic evil',
			}},
			regex=True)
	
	def processing(self):
		cols = ["acousticness", "danceability", "energy", "speechiness", "valence", "popularity"]
		self.df_x = self.data[cols].values
		self.df_y = self.data[['genres']].values
	
	def train(self):
		self.dt = DecisionTreeClassifier()
		self.dt.fit(self.df_x[:1000], self.df_y[:1000])
	
	def predict(self, track):
		return self.dt.predict(track)
