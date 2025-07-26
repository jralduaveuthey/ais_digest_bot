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

#### Required Services
- **Telegram Bot**: Create via [@BotFather](https://t.me/botfather)
- **OpenAI API**: Get API key from [OpenAI Platform](https://platform.openai.com/)
- **Mailgun**: Sign up at [Mailgun](https://www.mailgun.com/) for email functionality
- **Notion Integration**: Create an integration at [Notion Developers](https://developers.notion.com/)

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
   
   # Additional services
   MAILGUN_API_KEY=your_mailgun_api_key
   MAILGUN_DOMAIN=your_mailgun_domain
   NOTION_TOKEN_TASK_MASTER=your_notion_integration_token
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

- `/new` - Start new conversation
- `/agent` - Activate AI agent mode for task automation (email reminders, Notion tasks)
- `/email` - Send conversation summary via email (add text after command for additional info)
- `/retrieve` - Retrieve unprocessed content from content processor
- `/exam` - Check your understanding
- `/reflect` - Solo Reflection
- `/journal` - Private journaling
- `/journalgpt` - Journal with AI assistant
- `/stampy` - Ask question to Stampy (stampy.ai/chat/)   
- `/transcript` - Returns transcript or jina link

### Agent Mode (/agent)

The agent mode allows you to automate tasks using natural language. When activated, the bot can:

- **Send Email Reminders**: Automatically send emails with task descriptions
- **Create Notion Tasks**: Create tasks in your Notion database with due dates

Examples:
```
/agent
Send me an email to remind me to do the dishes before tomorrow
Create a Notion task to review the quarterly report
Email john@example.com about the meeting preparation
```

To exit agent mode, use the `/new` command.

**Note**: Agent mode always uses OpenAI's function calling capability with the model specified in `OPENAI_MODEL` (e.g., o4-mini), regardless of your default model provider configuration. Ensure your OpenAI API key is configured.

## Troubleshooting

### User Not Authorized
Add your Telegram username or ID to `USERS_ALLOWED` in the `.env` file

### Local Token Issues
Ensure `TELEGRAM_BOT_TOKEN_AIS_Digest_local` is set in `.env`

### AWS Parameter Store Fallback
Some parameters (like `AUX_USERNAME`) may still be fetched from AWS even when running locally if not defined in `.env`

### Agent Mode Issues
- Ensure `NOTION_TOKEN_TASK_MASTER` is set in `.env` for Notion integration
- Verify `MAILGUN_API_KEY` and `MAILGUN_DOMAIN` are configured for email functionality
- Agent mode always uses OpenAI API - ensure `OPENAI_API_KEY` is configured even if using Gemini as default