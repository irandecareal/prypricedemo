import boto3
import pandas as pd

def load_data(s3_bucket, s3_key):
    s3_client = boto3.client('s3')
    obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
    df = pd.read_csv(obj['Body'], sep=';')
    df['DATE'] = pd.to_datetime(df['DATE'])
    df.set_index('DATE', inplace=True)
    return df

def make_stationary(df):
    return df.dropna()

def split_data(df, train_size=0.8):
    ts = df['PRICE']
    train_size = int(len(ts) * 0.8)
    train, test = ts[:train_size], ts[train_size:]
    return train, test