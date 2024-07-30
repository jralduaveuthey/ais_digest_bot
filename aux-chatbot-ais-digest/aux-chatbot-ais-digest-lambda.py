import boto3
import json
import os
import random

#TODO: xaki; make sure that this is working and when i run it the bot actually replies to my normal telegram user
#-------------------AUX BOT SETTINGS-------------------#

MAIN_BOT_LAMBDA_FUNCTION_NAME_SSM_PATH = '/aux-chatbot-ais-digest/MAIN_BOT_LAMBDA_FUNCTION_NAME'
SIMULATION_CHAT_ID_SSM_PATH= '/aux-chatbot-ais-digest/SIMULATION_CHAT_ID'
AUX_USERNAME_SSM_PATH= '/aux-chatbot-ais-digest/AUX_USERNAME'

#--------------------------------------------------#

region = 'eu-central-1'
ssm = boto3.client('ssm', region_name=region)  # Adjust the region as needed.
s3_client = boto3.client('s3', region_name=region)
polly = boto3.client('polly')

def get_parameter(ssm, name):
    """Retrieve a parameter by name from AWS Systems Manager Parameter Store."""
    print(f"Getting parameter '{name}' from the AWS Parameter Store.")
    response = ssm.get_parameter(Name=name, WithDecryption=True)
    return response['Parameter']['Value']

def invoke_lambda(function_name, payload):
    """
    Invokes the specified Lambda function with the given payload.
    """
    lambda_client = boto3.client('lambda', region_name='eu-central-1')  # Replace with your region
    
    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )
    
    return json.loads(response['Payload'].read())

def simulate_telegram_message(chat_id, username, text):
    """
    Creates a simulated Telegram message payload.
    """
    return {
        "body": json.dumps({
            "message": {
                "chat": {
                    "id": chat_id,
                    "username": username
                },
                "text": text,
                "date": 1627776000,  # Example timestamp
                "from": {
                    "id": 123456789,
                    "is_bot": False,
                    "first_name": "Test",
                    "username": username
                },
                "message_id": 1
            }
        })
    }

def get_random_task():
    """
    Returns a random task for the bot to perform.
    """
    tasks = [
        "capital of spain?.",
        # "Remind the user to drink water and stay hydrated.",
        # "Share an interesting fact about space exploration.",
        # "Suggest a quick 5-minute exercise routine.",
        # "Provide a short meditation prompt for stress relief.",
    ]
    return random.choice(tasks)

def lambda_handler(event, context):
    """
    AWS Lambda function handler for aux-chatbot-ais-digest-lambda.
    """
    # Replace these with your actual values or retrieve from environment variables
    ssm = boto3.client('ssm', region_name=region)  # Adjust the region as needed.
    MAIN_BOT_LAMBDA_FUNCTION_NAME = get_parameter(ssm, MAIN_BOT_LAMBDA_FUNCTION_NAME_SSM_PATH)
    CHAT_ID = int(get_parameter(ssm, SIMULATION_CHAT_ID_SSM_PATH))
    USERNAME = get_parameter(ssm, AUX_USERNAME_SSM_PATH)

    task = get_random_task()
    payload = simulate_telegram_message(CHAT_ID, USERNAME, task)
    response = invoke_lambda(MAIN_BOT_LAMBDA_FUNCTION_NAME, payload)

    print(f"Task sent to main Lambda: {task}")
    print("Main Lambda function response:", response)

    return {
        'statusCode': 200,
        'body': json.dumps('aux-chatbot-ais-digest-lambda executed successfully')
    }

if __name__ == "__main__":
    # This block will only run when the script is executed locally, not in Lambda
    # Replace these with your actual values for local testing
    ssm = boto3.client('ssm', region_name=region)  # Adjust the region as needed.
    MAIN_BOT_LAMBDA_FUNCTION_NAME = get_parameter(ssm, MAIN_BOT_LAMBDA_FUNCTION_NAME_SSM_PATH)
    CHAT_ID = int(get_parameter(ssm, SIMULATION_CHAT_ID_SSM_PATH))
    USERNAME = get_parameter(ssm, AUX_USERNAME_SSM_PATH)

    task = get_random_task()
    payload = simulate_telegram_message(CHAT_ID, USERNAME, task)
    response = invoke_lambda(MAIN_BOT_LAMBDA_FUNCTION_NAME, payload)

    print(f"Task sent to main Lambda: {task}")
    print("Main Lambda function response:", response)