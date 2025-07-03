# AIS Digest Telegram Bot

A unified Telegram bot that works seamlessly in both local development and AWS Lambda environments.

## Overview

This bot uses a single `lambda-main.py` file that automatically detects its runtime environment and configures itself accordingly:
- **Local Development**: Uses local environment variables and a test bot token
- **AWS Lambda**: Uses AWS Systems Manager Parameter Store and production bot token

## Setup Instructions

### 1. Prerequisites
- Python 3.9+
- AWS CLI configured (for AWS deployment)
- Docker Desktop (for AWS deployment)
- Telegram bot tokens (one for local testing, one for production)

### 2. Initial Setup
```bash
# Clone the repository
git clone <repository-url>
cd ais_digest_bot

# Install dependencies
pipenv install
```

### 3. Configure Environment Variables

#### For Local Development
1. Copy the example environment file:
   ```bash
   cp chatbot-ais-digest/.env.example chatbot-ais-digest/.env
   ```

2. Edit `.env` and fill in your values:
   ```env
   # Local bot token
   TELEGRAM_BOT_TOKEN_AIS_Digest_local=your_local_bot_token
   
   # API Keys
   OPENAI_API_KEY=your_openai_key
   GOOGLE_API_KEY=your_google_key
   
   # S3 Bucket
   S3_BUCKET=your_bucket_name
   
   # Allowed users (comma-separated Telegram usernames or IDs)
   USERS_ALLOWED=username1,username2,12345678
   ```

#### For AWS Deployment
The `deployment.py` script automatically uploads environment variables to AWS Parameter Store.

### 4. Running Locally
```bash
cd chatbot-ais-digest
python lambda-main.py
```

The bot will:
- Detect it's running locally
- Load environment variables from `.env`
- Use the local bot token
- Log to both console and `lambda-main.log`

### 5. Deploying to AWS

1. Ensure Docker Desktop is running
2. Run the deployment script:
   ```bash
   cd chatbot-ais-digest
   python deployment.py
   ```

The deployment script will:
- Upload environment variables to AWS Parameter Store
- Build a Docker image
- Push to Amazon ECR
- Create/update the Lambda function

### 6. Setting up API Gateway
Follow the standard AWS API Gateway setup for Telegram webhooks:
1. Create an HTTP API in API Gateway
2. Set the route to `ANY /{proxy+}`
3. Configure Lambda integration with payload format version 1.0
4. Use the Lambda function URL for the Telegram webhook

## Environment Detection

The bot automatically detects its environment:

```python
def is_running_in_lambda():
    return bool(os.environ.get('AWS_LAMBDA_FUNCTION_NAME'))
```

Based on this detection:
- **Token Selection**: Uses `_local` suffix for local tokens
- **Parameter Retrieval**: Local uses `.env`, AWS uses Parameter Store
- **Logging**: Local logs to file, AWS logs to CloudWatch

## Bot Commands
- `/new` - Start a new conversation
- `/stampy` - Ask Stampy AI
- `/transcript` - Get transcript of linked content
- `/exam` - Test your understanding
- `/reflect` - Solo reflection mode
- `/journal` - Private journaling
- `/journalgpt` - Journal with AI assistance
- `/retrieve` - Retrieve unprocessed content

## Troubleshooting

### User Not Authorized
Add your Telegram username or ID to `USERS_ALLOWED` in the `.env` file

### Local Token Issues
Ensure `TELEGRAM_BOT_TOKEN_AIS_Digest_local` is set in `.env`

### AWS Parameter Store Fallback
Some parameters (like `AUX_USERNAME`) may still be fetched from AWS even when running locally if not defined in `.env`