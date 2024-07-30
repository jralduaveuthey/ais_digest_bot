import os
import subprocess
import boto3
import botocore
import time
import json
from dotenv import dotenv_values


parent_dir = os.path.dirname(os.path.abspath(__file__))
folder_name = os.path.basename(parent_dir)# Folder name and Lambda function name

# Load all environment variables from .env into a dictionary
env_vars = dotenv_values(f"{parent_dir}\\.env")

# Set up AWS ECR and Lambda clients
session = boto3.Session()
region = session.region_name
ecr_client = boto3.client('ecr')
lambda_client = boto3.client('lambda')

ssm_client = boto3.client('ssm', region_name=region)

# filtered_keys = {k:v for k, v in env_vars.items() if 'key' not in k.lower()} # Filter keys to exclude those containing 'key' in their name 
filtered_keys = env_vars

# Add each parameter to the AWS parameter store
for key, value in filtered_keys.items():
    ssm_client.put_parameter(
        Name=f'/{folder_name}/{key}',
        Value=value,
        Type='SecureString',
        Overwrite=True
    )
    print(f"Succesfully uploaded /{folder_name}/{key} to AWS Parameter store")
