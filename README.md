```shell script
conda create -n spotify python=3
source activate spotify
conda install flask
conda install pandas
conda install scikit-learn
conda install jupyter notebook
conda install requests
export FLASK_RUN_PORT=8000
FLASK_ENV=development flask run
```