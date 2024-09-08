import boto3
from botocore.exceptions import WaiterError
import logging
import sagemaker
from sagemaker.multidatamodel import MultiDataModel
from sagemaker import get_execution_role

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the SageMaker session
sagemaker_session = sagemaker.Session()

# Replace with your ECR image URI and role ARN
container_uri = '533266973518.dkr.ecr.us-east-1.amazonaws.com/price-prediction'
role_arn = 'arn:aws:iam::533266973518:role/rol-predict-price'
bucket = 'bk-price-prediction-data'
model_key = 'model'
# Define the S3 bucket prefix where the models are stored
s3_model_prefix = f's3://{bucket}/{model_key}/'


# Create the MultiDataModel object to manage multiple models from S3
multi_model = MultiDataModel(
    name='multi-model-endpoint',
    model_data_prefix=s3_model_prefix,  # S3 location of models
    image_uri=container_uri,  # Custom container URI
    role=role_arn,  # IAM role
    sagemaker_session=sagemaker_session
)

# Define the instance type and endpoint name
instance_type = 'ml.m5.large'
endpoint_name = 'multi-model-endpoint'

try:
    # Optionally, wait for the endpoint to be in service
    # Deploy the multi-model endpoint
    predictor = multi_model.deploy(
      initial_instance_count=1,
      instance_type=instance_type,
      endpoint_name=endpoint_name
    )

    print(f"Multi-Model Endpoint '{endpoint_name}' has been deployed on instance type '{instance_type}'.")
    logger.debug("Endpoint is in service and ready for use.")
except Exception as e:
    logger.error(f"Error deploying the Multi-Model Endpoint: {str(e)}")






