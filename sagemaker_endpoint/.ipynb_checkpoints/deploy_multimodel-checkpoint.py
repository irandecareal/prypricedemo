import boto3
from botocore.exceptions import WaiterError

sagemaker = boto3.client('sagemaker')

# Replace with your ECR image URI and role ARN
container_uri = '533266973518.dkr.ecr.us-east-1.amazonaws.com/price-prediction'
role_arn = 'arn:aws:iam::533266973518:role/rol-predict-price'
bucket = 'bk-price-prediction-data'
model_key = 'model'

# Create the model
responsea = sagemaker.create_model(
    ModelName='multi-model-endpoint',
    PrimaryContainer={
        'Image': container_uri,
        'ModelDataUrl': f's3://{bucket}/{model_key}/',
        'Mode': 'MultiModel'
    },
    ExecutionRoleArn=role_arn
)

print("Model created:", responsea['ModelArn'])

responseb = sagemaker.create_endpoint_config(
    EndpointConfigName='multi-model-endpoint-config',
    ProductionVariants=[
        {
            'VariantName': 'AllTraffic',
            'ModelName': 'multi-model-endpoint',
            'InstanceType': 'ml.m5.large',
            'InitialInstanceCount': 1,
        },
    ]
)

print("Endpoint configuration created:", responseb['EndpointConfigArn'])

response = sagemaker.create_endpoint(
    EndpointName='multi-model-endpoint',
    EndpointConfigName='multi-model-endpoint-config'
)

print("Endpoint creation initiated:", response['EndpointArn'])

try:
    # Optionally, wait for the endpoint to be in service
    waiter = sagemaker.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName='multi-model-endpoint')
    print("Endpoint is in service and ready for use.")
except WaiterError as e:
    print(f"Failed to bring the endpoint into service: {e}")
    print("Check the CloudWatch logs for more details.")






