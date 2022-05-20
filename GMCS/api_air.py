#!/usr/bin/env python
import flask
from flask import Flask, request

from prediction import main_p

app = Flask(__name__)

@app.route('/predict', methods = ['POST'])
def predict():
    if flask.request.method == "POST":
        json_ = request.json

        if json_ == None or json_ == "" or json_ == "null":
            response = "no data"
            code = 400
            print(f"code == {code}")
        else:
            response = main_p(json_)
            code = 200
            print(f"code == {code}")
        return response, code

if __name__ == '__main__':
    # serve(app, host = "xxx", port=5000)
    app.run(debug=True)
    
