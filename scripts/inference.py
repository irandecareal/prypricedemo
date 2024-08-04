import joblib
import pandas as pd
from flask import Flask, request, jsonify
import boto3
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize S3 client
s3 = boto3.client('s3')

# Define S3 bucket and model key
S3_BUCKET = 'bk-price-prediction-data'
MODEL_KEY = 'model/model.joblib'

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
    # Download the model from S3
    model_file = download_model_from_s3(S3_BUCKET, MODEL_KEY)
    logger.info(f'Model downloaded from S3: {MODEL_KEY}')
    # Load model from file-like object
    try:
        model = joblib.load(model_file)
        logger.info("Model loaded into memory")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return jsonify({'error': str(e)}), 500
    if request.content_type == 'application/json':
        input_json = request.get_json()
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
                # Set the DATE column as the index
                input_data.set_index('DATE', inplace=True)
            else:
                logger.error("Missing required columns 'date' and 'price'")
                return jsonify({'error': "Missing required columns 'date' and 'price'"}), 400

            logger.info(f"Processed input data: {input_data}")
            try:
                predictions = model.get_forecast(steps=len(input_data)).predicted_mean
                logger.info(f"Predictions: {predictions}")
                return jsonify(predictions.tolist())
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

