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
MODEL_PREFIX = 'model'

is_model_scaling = {
 'beef': True,
 'aceite_oliva': True,
 'ajo': False,
 'rice': True,
 'bread': True,
 'cebolla': True,
 'huevo': True,
 'jamon': True,
 'patatas': True,
 'pepino': True,
 'pimiento': True,
 'shrimp': False,
 'squid': False,
 'tomate': True,
}

model_minimo = {
 'beef': 2.155585,
 'aceite_oliva': 154.194138,
 'rice': 150.629633,
 'bread': 0.04,
 'cebolla': 8.622133,
 'huevo': 125.199775,
 'jamon': 210.886633,
 'patatas': 2.730000,
 'pepino': 37.207867,
 'pimiento': 50.000000,
 'tomate': 51.740523
}

model_maximo = {
 'beef': 3.458275,
 'aceite_oliva': 735.892000,
 'rice': 877.082913,
 'bread': 1.48,
 'cebolla': 75.919400,
 'huevo': 308.611574,
 'jamon': 366.069492,
 'patatas': 58.510000,
 'pepino': 184.978120,
 'pimiento': 213.519940,
 'tomate': 226.323017
}

model_divider = {
 'beef': 1,
 'aceite_oliva': 100,
 'rice': 1000,
 'bread': 1,
 'cebolla': 100,
 'huevo': 100,
 'jamon': 100,
 'patatas': 100,
 'pepino': 100,
 'pimiento': 100,
 'tomate': 100
}

def downscaling_prediction(x_min, x_max, predicted_price_scaled, divider):
    # Downscaling the prediction
    # Manually downscale using the inverse of Min-Max scaling
    predicted_price = predicted_price_scaled * (x_max - x_min) + x_min
    logger.info(predicted_price)
    return predicted_price/divider

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
                if input_data['PRICE'].eq('').any():
                    logger.info('empty price')
                    input_data = input_data.drop('PRICE', axis=1)
                    input_data.rename(columns={"DATE": "ds"}, inplace=True)
                else:
                    columns_parse = {"DATE": "ds", "PRICE": "y"}
                    input_data.rename(columns=columns_parse, inplace=True)

                logger.info('before predict')
                logger.info(f"input_data {input_data}")
                logger.info(f"dataDATE {data['DATE'][0]}")
                # date_parse = pd.to_datetime(data['DATE'][0])
                # logger.info(f"data['PRICE'] {data['PRICE']}")
                # amount_parse = data['PRICE']
                # dfinput = pd.DataFrame({'ds': [date_parse], 'y': amount_parse})
                # forecast = model.predict(dfinput)
                # logger.info(f"forecast {forecast}")
                forecast = model.predict(input_data)
                logger.info(f"forecast {forecast}")
                predicted_price = forecast['yhat'].values[0]
                logger.info(f"Predicted price for {input_data['ds']}: {predicted_price}")
                if is_model_scaling[ingredient]:
                    amount_min = model_minimo[ingredient]
                    amount_max = model_maximo[ingredient]
                    amount_divider = model_divider[ingredient]
                    price_parse = downscaling_prediction(amount_min, amount_max, predicted_price, amount_divider)
                    return jsonify({"price_forecasted": price_parse})
                return jsonify({"price_forecasted": predicted_price})
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

