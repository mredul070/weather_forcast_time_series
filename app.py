from flask import Flask 
from flask import request

from inference import get_prediction

app = Flask(__name__)

@app.route('/')
def index():
    return "Hi"

@app.route('/predict_temp', methods = ['GET'])
def predict_weather():
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
            return {"res" : "This weather is too extreme for Dhaka city Please Recheck the Inputted Values"}
    # get prediction from the model
    res = get_prediction(formatted_temp)[0][0]
    return {"res" : str(res)}


if __name__ == '__main__':
    app.run(debug=True)