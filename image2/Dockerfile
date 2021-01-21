#app2 : predictor
FROM python:3.8
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1                                     
CMD ["python", "app.py"]

