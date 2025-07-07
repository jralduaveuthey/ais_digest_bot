# Model Configuration System

This document explains how to configure the AI model provider and models for the chatbot.

## Overview

The chatbot now supports dynamic model configuration through AWS Parameter Store. This allows you to change the model provider and specific models without redeploying the Lambda function.

## Configuration Variables

Add these variables to your `.env` file:

```bash
# Model Configuration
MODEL_PROVIDER=gemini          # Options: "gemini" or "openai"
GEMINI_MODEL=gemini-2.5-flash  # Gemini model to use
OPENAI_MODEL=gpt-4o-mini       # OpenAI model to use
```

## Supported Providers

### Gemini (Google)
- **Provider**: `gemini`
- **Default Model**: `gemini-2.5-flash`
- **Context Window**: 1,048,576 tokens
- **API Key Required**: `GOOGLE_API_KEY`

### OpenAI
- **Provider**: `openai`
- **Default Model**: `gpt-4o-mini`
- **Context Window**: 128,000 tokens
- **API Key Required**: `OPENAI_API_KEY`

## How It Works

1. **Local Development**: Model configuration is loaded from environment variables (`.env` file)
2. **AWS Lambda**: Model configuration is loaded from AWS Parameter Store
3. **Deployment**: The `deployment.py` script automatically uploads all `.env` variables to Parameter Store

## Deployment Process

1. Add the model configuration variables to your `.env` file
2. Run the deployment script: `python deployment.py`
3. The script will automatically upload the new configuration to AWS Parameter Store
4. The Lambda function will use the new configuration on the next invocation

## Changing Models Without Redeployment

To change the model configuration without redeploying:

1. Update the parameters directly in AWS Parameter Store:
   - `/chatbot-ais-digest/MODEL_PROVIDER`
   - `/chatbot-ais-digest/GEMINI_MODEL`
   - `/chatbot-ais-digest/OPENAI_MODEL`

2. The changes will take effect on the next Lambda invocation

## Fallback Behavior

If the model configuration cannot be loaded from Parameter Store, the system will fall back to these defaults:
- **Provider**: `gemini`
- **Gemini Model**: `gemini-2.5-flash`
- **OpenAI Model**: `gpt-4o-mini`

## Example .env File

```bash
# Model Configuration
MODEL_PROVIDER=gemini
GEMINI_MODEL=gemini-2.0-flash
OPENAI_MODEL=gpt-4o-mini

# API Keys (required for the respective providers)
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key

# ... other configuration variables
```

## Troubleshooting

- **Error loading model config**: Check that the Parameter Store parameters exist and are accessible
- **Unsupported provider**: Ensure `MODEL_PROVIDER` is either `gemini` or `openai`
- **API errors**: Verify that the correct API key is configured for the selected provider 