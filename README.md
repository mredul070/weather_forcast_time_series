# Weather Forcast Time Series
***This repository contains code to predict weather condition based on time series using various LSTM.***


## Assumptions

- User will input the previous temperature of certain number of days sequentially, based on which we will predict next day's temperature
- Here the mean temperature is predicted for a particular day

## Why LSTM
- LSTM networks are well-suited to classifying, processing and making predictions based on time series data, since there can be lags of unknown duration between important events in a time series. LSTMs were developed to deal with the vanishing gradient problem that can be encountered when training traditional RNNs

- Using LSTM, time series forecasting models can predict future values based on previous, sequential data. This provides greater accuracy for demand forecasters which results in better decision making for the business.

****
# Project Setup/Installation
*Please install python for this project in your pc.*

- ### Step 1 : Create a veritual Environment
Create using any software of your choice. If you use Anaconda use the following command for creating.
```
conda create -n env_name python=3.8
```
python 3.8 is the preferred version for this project.

- ### Step 2 : Download or clone code repository
Now download the code base in local repository. You directly download download or you can use git to download the codes if you have git cli installed in your computer.
```
git clone https://github.com/mredul070/weather_forcast_time_series.git
``` 
Then take your terminal to required directory.

- ### Step 3 : Dependency Installation
To install all dependency required for this project use the following command.
```
pip install -r requirements.txt
```

- ### Step 4 : Run the Project
To run the project use the following command
```
python app.py
```
This command will launch a development server in your computer at the following ip **127.0.0.1:5000**

****
## API Endpoint
- path : **127.0.0.1:5000/predict_temp**
- method : GET
- querystring parameter : temp
    - example : temp=[20,25,22,23,24]
N.B: Please use this fixed for calling this API.

The values define average temperature for 5 sequencial days which are comma seperated. The value is the mean temperature of the earliest day and so on.

- Reponse : {"res": value}
You will receive the mean temperature for the next day in the response.

N.B: As the model was trained on the Dhaka city any extreme temperature will be discounted. API response will notify if any extreme case is Detected.

## Config 
In the project repository there is a config.py file which sets up the parameter for both model training and inference. This is simplify the training and inference process and create a single point of control.

### Time Steps
This is define how many previous data is require to predict the temperature of the next day. You can change this to any day. The model training and inference will update autometically. But then you will have provice that many number of days in the querystring to get the result.

    
