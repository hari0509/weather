from flask import Flask, jsonify
import pandas as pd
import requests
import joblib

app = Flask(__name__)

@app.route('/p', methods=['GET'])
def p():
    new_api_url = 'https://api.thingspeak.com/channels/2162931/feeds.json?results=1'
    new_response = requests.get(new_api_url)

    data = new_response.json()
    field1 = [entry['field1'] for entry in data['feeds']]
    field2 = [entry['field2'] for entry in data['feeds']]
    field3 = [entry['field3'] for entry in data['feeds']]
    field4 = [entry['field4'] for entry in data['feeds']]
    new_data = pd.DataFrame({
        'precipitation': field1,
        'temp_max': field2,
        'temp_min': field3,
        'wind': field4
    })
    loaded_model = joblib.load('trained_model.joblib')
    new_predictions = loaded_model.predict(new_data)
    return jsonify({'predictions': new_predictions.tolist()})

if __name__ == '__main__':
    app.run()

