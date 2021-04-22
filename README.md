# spotify backend

this is the repo for the backend portion of this [Spotify Evaluation App](https://github.com/iwasnevergivenaname/spotify-evaluation)

this application uses [Pandas](https://pandas.pydata.org/) to access and manipulate data on Spotify genres, sourced from [this Kaggle Notebook](https://www.kaggle.com/yamaerenay/spotify-dataset-19212020-160k-tracks). 
after that data has been consolidated into something we can easily use, [Scikit Learn](https://scikit-learn.org/stable/) provides
a DecisionTreeClassifier that we train with said data. after it has been trained, we can input sample data from a song, 
and receive a prediction from the model as to where that song falls in terms of our moral alignments that we previously assigned to our big data set.

## to start your server
in your terminal run
```shell script
conda create -n spotify python=3
source activate spotify
conda install flask
conda install pandas
conda install scikit-learn
conda install jupyter notebook
conda install numpy
conda install requests
export FLASK_RUN_PORT=8000
FLASK_ENV=development flask run
```

## to play around with the data
you can open a jupyter notebook

```shell script
cd src
jupyter notebook
```
a browser window should automatically open, with your notebook started.

create a new Python3 notebook in the services directory, and import 
```python
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
```
and then read data_by_genres.csv (please make sure your path is correct)
```python
df = pd.read_csv('../data/data_by_genres.csv')
```
