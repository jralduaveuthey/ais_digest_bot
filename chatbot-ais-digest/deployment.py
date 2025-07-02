import os
import subprocess
import boto3
import botocore
import time
import json
from dotenv import dotenv_values

def create_repository(ecr_client, repository_name):
    try:
        response = ecr_client.create_repository(
            repositoryName=repository_name
        )
        return response['repository']['repositoryUri']
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        print(f"Repository '{repository_name}' already exists.")

parent_dir = os.path.dirname(os.path.abspath(__file__))
folder_name = os.path.basename(parent_dir)# Folder name and Lambda function name

# Load all environment variables from .env into a dictionary
env_vars = dotenv_values(os.path.join(parent_dir, ".env"))

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

# Change to the folder where this script is located
os.chdir(parent_dir)

# Go one directory up for the local_test_bot.py file
parent_directory = os.path.dirname(parent_dir)
local_test_bot_path = os.path.join(parent_directory, 'local_test_bot.py')

# Path for lambda-main.py remains same
lambda_main_path = os.path.join(parent_dir, 'lambda-main.py')

# Open the local_test_bot.py in read mode
with open(local_test_bot_path, 'r') as local_test_bot_file:
    # Open the lambda-main.py in write mode
    with open(lambda_main_path, 'w') as lambda_main_file:
        for line in local_test_bot_file:
            # If the line starts with ##################, stop copying
            if line.startswith('##################'):
                break
            # Otherwise, write the line to lambda-main.py
            lambda_main_file.write(line)

# Just to be sure, replace '_local' with ''
with open(lambda_main_path, 'r') as file:
    file_data = file.read()
file_data = file_data.replace('_local', '')
with open(lambda_main_path, 'w') as file:
    file.write(file_data)

# Build the Docker image with a tag
tag = "latest"
print("Building docker image locally...")

# Ensure buildx is available and set as default
buildx_setup_cmd = "docker buildx create --use"
try:
    subprocess.run(buildx_setup_cmd, shell=True, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
except Exception as e:
    print("Buildx setup warning: ", str(e))

# Build with buildx for proper platform support
docker_build_cmd = f"docker buildx build --platform linux/amd64 --tag {folder_name}:{tag} --load ."
try:
    result = subprocess.run(docker_build_cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
except Exception as e:
    print("An error occurred: ", str(e))

print("stdout:", result.stdout)
print("stderr:", result.stderr)
result.check_returncode()

# Authenticate Docker to AWS ECR
ecr_repository_name = folder_name
try:
    response = ecr_client.describe_repositories(repositoryNames=[ecr_repository_name])
except ecr_client.exceptions.RepositoryNotFoundException:
    print(f"The ECR repository with name '{ecr_repository_name}' does not exist in AWS. It will be created now")
    create_repository(ecr_client, ecr_repository_name)
    response = ecr_client.describe_repositories(repositoryNames=[ecr_repository_name])

registry_id = response['repositories'][0]['registryId']
registry_uri = response['repositories'][0]['repositoryUri']

login_cmd = ecr_client.get_authorization_token(registryIds=[registry_id])['authorizationData'][0]['proxyEndpoint']
try:
    subprocess.run(f"aws ecr get-login-password --region {region} | docker login --username AWS --password-stdin {login_cmd}", shell=True, check=True)
except subprocess.CalledProcessError as e:
    print("Subprocess returned an error:", e)

# Tag and push the image to AWS ECR
docker_tag_cmd = f"docker tag {folder_name}:{tag} {registry_uri}:{tag}"
subprocess.run(docker_tag_cmd, shell=True, check=True)

docker_push_cmd = f"docker push {registry_uri}:{tag}"
subprocess.run(docker_push_cmd, shell=True, check=True)

# Update (or create if it does not exist) the Lambda function
lambda_function_name = folder_name

try:
    lambda_client.get_function(FunctionName=lambda_function_name)
    update = True
except lambda_client.exceptions.ResourceNotFoundException:
    update = False

sts_client = boto3.client('sts')
account_id = sts_client.get_caller_identity()["Account"]
iam = boto3.client('iam')

role_name = f'{lambda_function_name}_execution_role'
assume_role_policy_document = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "lambda.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}

try:
    # Check if the role exists
    response = iam.get_role(RoleName=role_name)
    print(f"Role '{role_name} already exists.")
except iam.exceptions.NoSuchEntityException:
    # If role does not exist, create it
    response = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(assume_role_policy_document)
    )
    print(f"Role '{role_name}' created.")

policies = [
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole",
    "arn:aws:iam::aws:policy/AmazonPollyFullAccess", 
    "arn:aws:iam::aws:policy/AmazonTranscribeFullAccess",
    "arn:aws:iam::aws:policy/AmazonSSMReadOnlyAccess"
]

# Custom policy for content processor access (if it exists)
custom_policy_arn = f"arn:aws:iam::{account_id}:policy/TelegramBotContentProcessorAccess"
try:
    iam.get_policy(PolicyArn=custom_policy_arn)
    policies.append(custom_policy_arn)
    print(f"Added custom policy: TelegramBotContentProcessorAccess")
except iam.exceptions.NoSuchEntityException:
    print(f"Custom policy TelegramBotContentProcessorAccess not found, skipping...")

# Add inline policy for s3:HeadBucket and s3:ListBucket
# Check if S3_BUCKET is defined in environment variables
if 'S3_BUCKET' not in env_vars or not env_vars['S3_BUCKET']:
    print("Error: S3_BUCKET is required in the .env file but was not found or is empty.")
    print("Please add S3_BUCKET=your-bucket-name to your .env file and try again.")
    exit(1)

s3_access_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:ListBucket",
                "s3:Get*",
                "s3:PutObject"
            ],
            "Resource": [
                f"arn:aws:s3:::{env_vars['S3_BUCKET']}",
                f"arn:aws:s3:::{env_vars['S3_BUCKET']}/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:HeadBucket"
            ],
            "Resource": [
                "*"
            ]
        }
    ]
}

# Convert policy to JSON string
s3_access_policy_json = json.dumps(s3_access_policy)

# Attach the policies if they are not already attached
for policy_arn in policies:
    attached_policies = iam.list_attached_role_policies(
        RoleName=role_name)['AttachedPolicies']

    if not any(policy['PolicyArn'] == policy_arn for policy in attached_policies):
        response = iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn=policy_arn
        )
        print(f"Policy {policy_arn} attached to role.")

# Put the inline policy
response = iam.put_role_policy(
    RoleName=role_name,
    PolicyName='S3AccessPolicy',
    PolicyDocument=s3_access_policy_json
)

# To ensure that the role propagation finishes before creating a function
time.sleep(10)

lambda_Timeout=600 #seconds
lambda_MemorySize=1280 #Mb
max_retries = 5
retry_delay = 10 # delay in seconds

if update:
    # update function code
    for i in range(max_retries):
        try:
            print("Trying to update the code of the lambda")
            response = lambda_client.update_function_code(
                FunctionName=lambda_function_name,
                ImageUri=f"{registry_uri}:{tag}"
            )
            print("Succesfully updated the code of the lambda")
            break
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'ResourceConflictException' and i < max_retries - 1:
                time.sleep(retry_delay) # wait before retrying
            else:
                raise

    # update function configuration
    for i in range(max_retries):
        try:
            print("Trying to update the configuration of the lambda")
            response = lambda_client.update_function_configuration(
                FunctionName=lambda_function_name,
                Timeout=lambda_Timeout, 
                MemorySize=lambda_MemorySize 
            )
            print("Succesfully updated the configuration of the lambda")
            break
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'ResourceConflictException' and i < max_retries - 1:
                time.sleep(retry_delay) # wait before retrying
            else:
                raise
else:
    response = lambda_client.create_function(
        FunctionName=lambda_function_name,
        Role=f'arn:aws:iam::{account_id}:role/{role_name}',
        Code={
            'ImageUri': f"{registry_uri}:{tag}"
        },
        PackageType='Image',
        Timeout=lambda_Timeout,
        MemorySize=lambda_MemorySize
    )

print(f"Lambda function {lambda_function_name} updated with image {registry_uri}:{tag}")