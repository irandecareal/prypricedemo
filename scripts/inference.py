import joblib
import pandas as pd
from flask import Flask, request, jsonify
import boto3
import io
import logging
from prophet import Prophet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize S3 client
s3 = boto3.client('s3')

# Define S3 bucket and model key
S3_BUCKET = 'bk-price-prediction-data'
MODEL_PREFIX = 'model'

def download_model_from_s3(bucket, key):
    """Download a file from S3 and return as an in-memory file-like object"""
    temp_file = io.BytesIO()
    s3.download_fileobj(bucket, key, temp_file)
    temp_file.seek(0)  # Reset file pointer to the beginning
    return temp_file

@app.route('/ping', methods=['GET'])
def ping():
    logger.info("Ping received")
    return jsonify({'status': 'healthy'}), 200

@app.route('/invocations', methods=['POST'])
def predict():
    logger.info("Invocation request received")
    # Log the request URL, headers, and data
    logger.info(f"Request URL: {request.url}")
    logger.info(f"Request headers: {request.headers}")
    logger.info(f"Request data: {request.data}")
    input_json = request.get_json()
    ingredient = input_json['ingredient']
    if not ingredient:
        logger.error("Missing 'ingredient' parameter")
        return jsonify({'error': "Missing 'ingredient' parameter"}), 400
    specific_date = input_json['date_forescast']
    if not specific_date:
        logger.error("Missing 'date_forescast' parameter")
        return jsonify({'error': "Missing 'date_forescast' parameter"}), 400        

    # Download the model from S3
    model_key = f"{MODEL_PREFIX}/model_{ingredient}.joblib"
    model_file = download_model_from_s3(S3_BUCKET, model_key)
    logger.info(f'Model downloaded from S3: {model_key}')
    # Load model from file-like object
    try:
        model = joblib.load(model_file)
        logger.info("Model loaded into memory")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return jsonify({'error': str(e)}), 500
    if request.content_type == 'application/json':
        # Ensure input_json has the key 'data'
        if 'data' in input_json:
            data = input_json['data']
            logger.info(f"input JSON DATA : {data}")
            input_data = pd.DataFrame(data)
            logger.info(f"input data: {input_data}")
            # Data preprocessing steps
            # Ensure the DataFrame has 'date' and 'price' columns
            if 'DATE' in input_data.columns and 'PRICE' in input_data.columns:
                logger.info(f"input data DATE: {input_data['DATE']}")
                input_data['DATE'] = pd.to_datetime(input_data['DATE'])
            else:
                logger.error("Missing required columns 'date' and 'price'")
                return jsonify({'error': "Missing required columns 'date' and 'price'"}), 400

            logger.info(f"Processed input data: {input_data}")
            try:
                input_data.rename(columns={"DATE": "ds", "PRICE": "y"}, inplace=True)
                future = model.make_future_dataframe(periods=6000, freq='D')
                future = pd.concat([future, input_data[['ds', 'y']]], ignore_index=True)
                forecast = model.predict(future)
                prediction_for_specific_date = forecast[forecast['ds'] == specific_date]['yhat'].values[0]
                logger.info(f"Predictions: {prediction_for_specific_date}")
                return jsonify({"price_forecasted": prediction_for_specific_date})
            except Exception as e:
                logger.error(f"Error making predictions: {e}")
                return jsonify({'error': str(e)}), 500
        else:
            logger.error("Missing 'data' key in JSON input")
            return jsonify({'error': "Missing 'data' key in JSON input"}), 400
    else:
        logger.error("Unsupported content type")
        return jsonify({'error': 'Unsupported content type'}), 415


if __name__ == '__main__':
    logger.info("Starting server")
    app.run(host='0.0.0.0', port=8080)

