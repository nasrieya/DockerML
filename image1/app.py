import requests
from flask import Flask, jsonify, request
import json
import base64


app = Flask(__name__)
  
@app.route("/pred", methods=["POST"])
def consume():
	url = 'http://0.0.0.0:4200/api/train'
	
	fich = open('disco.00004.wav', 'rb') 

	
	data_encode = base64.b64encode(fich.read())

	
	headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
	#r = requests.post(url, data=base64_str, headers=headers)
	r = requests.post(url, json={"audio": data_encode.decode("utf-8")}, headers=headers)
	
	print(r, r.text)
	return (r.text)
		  
    

if __name__ == '__main__':
   app.run(debug=True, port=5000, host='0.0.0.0', threaded=True)
   
   
