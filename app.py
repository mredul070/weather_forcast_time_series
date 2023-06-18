import json
from flask import Flask 
from flask import request

from inference import get_prediction

app = Flask(__name__)

@app.route('/')
def index():
    return "Hi"

@app.route('/predict_temp', methods = ['GET'])
def predict_weather():
    try:
        # get temperatues value
        temperatures = request.args.get('temp')
        # remove parenthesis
        temperatures = temperatures[1:-1]
        formatted_temp = []
        # format and check provided temperatures
        for temp in temperatures.split(','):
            if 13.6 < float(temp) < 34.0:
                formatted_temp.append([float(temp)])
            else:
                return json.dumps({
                    "statusCode" : 200,
                    "res" : "This weather is too extreme for Dhaka city Please Recheck the Inputted Values"
                    })
        # get prediction from the model
        res = get_prediction(formatted_temp)[0][0]
        return json.dumps({
            "statusCode" : 200,
            "res" : str(res)
            })
    except Exception as e:
        return json.dumps({
            "statusCode" : 500,
            "key" : "Something Went Wrong"
        })


if __name__ == '__main__':
    app.run(debug=True)