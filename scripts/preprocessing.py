import boto3
import pandas as pd
import io
import joblib
import tarfile
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data_csv(s3_bucket, s3_key, separator):
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
    df = pd.read_csv(obj['Body'], sep=separator)
    return df

def make_stationary(df):
    return df.dropna()

def split_data(df, train_size=0.8):
    train_size = int(len(df) * 0.8)
    train = df[:train_size]
    test = df[train_size:]
    return train, test

def save_model_local(model, name_model):
    model_file_path = f'../model/{name_model}.joblib'
    # Path to the output tar.gz file
    output_tar_path = f'../model/{name_model}.tar.gz'

    joblib.dump(model, model_file_path)
    # Create a tar.gz archive
    with tarfile.open(output_tar_path, 'w:gz') as tar:
        tar.add(model_file_path, arcname=f'${name_model}.joblib')
    print(f"Created {output_tar_path} containing {model_file_path}")

def upload_model_s3(name_model, bucket):
    model_file_path = f'../model/{name_model}.joblib'
    output_tar_path = f'../model/{name_model}.tar.gz'
    s3_client = boto3.client('s3')
    s3_client.upload_file(output_tar_path, bucket, f'model/{name_model}.tar.gz')
    s3_client.upload_file(model_file_path, bucket, f'model/{name_model}.joblib')
    print('upload s3')

def mape(true, pred):
    return np.mean(np.abs((true - pred) / true)) * 100
    
def calculate_metrics(true_values, predictions):
    # Calcular las métricas
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = mse**0.5
    mape_value = mape(true_values, predictions)

    # Crear una tabla con los resultados
    results = [
        ["MAE", mae],
        ["MSE", mse],
        ["RMSE", rmse],
        ["MAPE", mape_value]
    ]
    return results
