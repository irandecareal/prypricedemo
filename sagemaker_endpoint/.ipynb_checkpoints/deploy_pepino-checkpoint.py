import boto3
from sagemaker.sklearn.model import SKLearnModel

s3_client = boto3.client('s3')
bucket = 'bk-price-prediction-data'
model_key = 'model/model_pepino.tar.gz'
role = 'rol-predict-price'
# Specify the desired framework and Python versions
framework_version = '0.23-1'  # Example: Scikit-learn version 0.23-1
py_version = 'py3'  # Python 3
image_uri = '533266973518.dkr.ecr.us-east-1.amazonaws.com/price-prediction'


# Deploy the model
model = SKLearnModel(
    model_data=f's3://{bucket}/{model_key}',
    role=role,
    image_uri=image_uri,
    entry_point='../scripts/inference.py',
    framework_version=framework_version,
    py_version=py_version
)

predictor = model.deploy(instance_type='ml.m4.xlarge', initial_instance_count=1,
                         endpoint_name='price-prediction-endpoint-pepino')
