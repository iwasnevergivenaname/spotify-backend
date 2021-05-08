FROM frolvlad/alpine-miniconda3

WORKDIR /app
COPY ../requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . .

RUN pip3 install -r requirements.txt

CMD ["python3", "-m", "flask", "run"]
#CMD ["python3", "-m", "flask", "run", "--host", "0.0.0.0:5500"]
#CMD ["python3", "wsgi.py"]