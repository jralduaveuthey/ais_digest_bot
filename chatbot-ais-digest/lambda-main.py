import io
import time
import boto3
import json
import logging
import requests
from datetime import datetime
import hashlib
from typing import List, Dict, Any, Union, Tuple, Optional, Callable
from contextlib import closing
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from bs4 import BeautifulSoup
import logging
import tiktoken
import openai
import google.generativeai as genai
import os
from notion_client import Client as NotionClient

# Environment detection function (defined early for logging setup)
def is_running_in_lambda():
    """Detect if code is running in AWS Lambda environment."""
    return bool(os.environ.get('AWS_LAMBDA_FUNCTION_NAME'))

# Configure logging based on environment
if is_running_in_lambda():
    # Lambda environment - log to CloudWatch
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Only use console output, which goes to CloudWatch
        ]
    )
else:
    # Local environment - log to both console and file
    log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lambda-main.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler(log_file)  # File output
        ]
    )

logger = logging.getLogger(__name__)

#-------------------MODEL PROVIDER SETTINGS-------------------#
# Model provider configuration - these will be loaded from Parameter Store
# Default fallback values (will be overridden by Parameter Store values)
DEFAULT_MODEL_PROVIDER = "gemini"  # Options: "gemini", "openai"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

# These will be set after loading from Parameter Store
MODEL_PROVIDER = None
GEMINI_MODEL = None
OPENAI_MODEL = None
DEFAULT_MODEL = None
MAX_TOKENS = None

DEFAULT_ENCODING = "cl100k_base"  # This is used by gpt-3.5-turbo and gpt-4

def initialize_model_config(model_provider, gemini_model, openai_model):
    """Initialize model configuration based on Parameter Store values."""
    global MODEL_PROVIDER, GEMINI_MODEL, OPENAI_MODEL, DEFAULT_MODEL, MAX_TOKENS
    
    MODEL_PROVIDER = model_provider
    GEMINI_MODEL = gemini_model
    OPENAI_MODEL = openai_model
    
    # Set default model and max tokens based on provider
    if MODEL_PROVIDER == "gemini":
        DEFAULT_MODEL = GEMINI_MODEL
        MAX_TOKENS = 1048576  # Gemini 2.0 Flash has 1M token context window
    elif MODEL_PROVIDER == "openai":
        DEFAULT_MODEL = OPENAI_MODEL
        MAX_TOKENS = 128000  # GPT-4o-mini has 128K token context window
    else:
        raise ValueError(f"Unsupported model provider: {MODEL_PROVIDER}")
    
    logger.info(f"Model configuration initialized: Provider={MODEL_PROVIDER}, Model={DEFAULT_MODEL}, Max Tokens={MAX_TOKENS}")
#--------------------------------------------------#

#-------------------BOT SETTINGS-------------------#
"""
List of bot commands:
/new - Start new conversation
/stampy - Ask question to Stampy (stampy.ai/chat/)
/transcript - Returns transcript or jina link
/exam - Check your understanding
/reflect - Solo Reflection
/journal - Private journaling
/journalgpt - Journal with AI assistant
/retrieve - Retrieve unprocessed content from content processor
/email - Send conversation summary via email (add text after command for additional info)
/agent - Activate AI agent mode for task automation (email, Notion tasks)
"""

prompt_template = """The following is a friendly conversation between a human and an AI. 
    The AI is talkative and provides lots of specific details from its context. 
    If the AI does not know the answer to a question, it truthfully says it does not know.
    Here is the history of the conversation that must be checked before answering any question: 
    {conversation_history}
    Now, the conversation continues:
    Human: {input}
    AI:"""

# Dynamically set the token path based on environment
if is_running_in_lambda():
    TELEGRAM_BOT_TOKEN_SSM_PATH = '/chatbot-ais-digest/TELEGRAM_BOT_TOKEN_AIS_Digest'
    logger.info("Running in AWS Lambda environment - using production token")
else:
    TELEGRAM_BOT_TOKEN_SSM_PATH = '/chatbot-ais-digest/TELEGRAM_BOT_TOKEN_AIS_Digest_local'
    logger.info("Running locally - using local token")
OPENAI_API_KEY_SSM_PATH = '/chatbot-ais-digest/OPENAI_API_KEY'
GOOGLE_API_KEY_SSM_PATH = '/chatbot-ais-digest/GOOGLE_API_KEY'
S3_BUCKET_SSM_PATH = '/chatbot-ais-digest/S3_BUCKET'
USERS_ALLOWED_SSM_PATH = '/chatbot-ais-digest/USERS_ALLOWED'
RAPIDAPI_KEY_SSM_PATH = '/chatbot-ais-digest/RAPIDAPI_KEY'
AUX_USERNAME_SSM_PATH = '/aux-chatbot-ais-digest/AUX_USERNAME'
MAILGUN_API_KEY_SSM_PATH = '/chatbot-ais-digest/MAILGUN_API_KEY'
MAILGUN_DOMAIN_SSM_PATH = '/chatbot-ais-digest/MAILGUN_DOMAIN'
MODEL_PROVIDER_SSM_PATH = '/chatbot-ais-digest/MODEL_PROVIDER'
GEMINI_MODEL_SSM_PATH = '/chatbot-ais-digest/GEMINI_MODEL'
OPENAI_MODEL_SSM_PATH = '/chatbot-ais-digest/OPENAI_MODEL'
NOTION_TOKEN_SSM_PATH = '/chatbot-ais-digest/NOTION_TOKEN_TASK_MASTER'

# Notion configuration
NOTION_TASKS_DATABASE_ID = "225fcfd1de9d800eae33de09d456e1d2"
NOTION_JAIME_USER_ID = "104d872b-594c-81d1-a431-0002006e3bbe"

# Content processor settings
CONTENT_PROCESSOR_BUCKET = "content-processor-110199781938"
CONTENT_PROCESSOR_STATE_FILE = "processing-state/state.json"

voice_name="Joanna" #"Amy"#"Geraint"#"Joanna"  #to check other premade voices go to https://docs.aws.amazon.com/polly/latest/dg/voicelist.html
MAX_RESPONSE_LENGTH_AUDIO = 3000  # Adjust as needed for Polly's limitations

# List of allowed domains for content extraction
FORBIDDEN_DOMAINS = [
    # 'www.lesswrong.com',
    # 'https://forum.effectivealtruism.org/',
    # Add more domains as needed
]

BOT_AUDIO_RESPONSE = False
#--------------------------------------------------#

region = 'eu-central-1'

# Initialize AWS clients conditionally based on environment
if is_running_in_lambda():
    ssm = boto3.client('ssm', region_name=region)
    s3_client = boto3.client('s3', region_name=region)
    polly = boto3.client('polly')
else:
    # For local development, these will be initialized in main() after loading .env
    ssm = None
    s3_client = None
    polly = None

JOURNAL_FILE = "userlogs/journal.json"
JOURNALGPT_FILE = "userlogs/journalgpt.json"

# Custom API Client Wrapper
class LLMClient:
    def __init__(self, provider: str, api_key: str, model_name: str = DEFAULT_MODEL):
        self.provider = provider.lower()
        self.api_key = api_key
        self.model_name = model_name
        
        if self.provider == "openai":
            openai.api_key = api_key
        elif self.provider == "gemini":
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
    def generate(self, messages: List[Dict[str, str]], max_retries: int = 1, functions: List[Dict] = None, function_call: str = "auto") -> Union[str, Dict]:
        """Generate a response from the specified API, optionally with function calling."""
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    kwargs = {
                        "model": self.model_name,
                        "messages": messages
                    }
                    if functions:
                        kwargs["functions"] = functions
                        kwargs["function_call"] = function_call
                    
                    response = openai.ChatCompletion.create(**kwargs)
                    
                    # Check if a function was called
                    message = response.choices[0].message
                    if hasattr(message, 'function_call') and message.function_call:
                        return {
                            "type": "function_call",
                            "function_call": message.function_call
                        }
                    return message.content
                elif self.provider == "gemini":
                    # Convert OpenAI-style messages to Gemini format
                    prompt_parts = []
                    for message in messages:
                        role = message.get("role", "user")
                        content = message.get("content", "")
                        if role == "system":
                            prompt_parts.append(f"System: {content}")
                        elif role == "user":
                            prompt_parts.append(f"Human: {content}")
                        elif role == "assistant":
                            prompt_parts.append(f"Assistant: {content}")
                    
                    full_prompt = "\n".join(prompt_parts)
                    response = self.model.generate_content(full_prompt)
                    return response.text
                else:
                    raise ValueError(f"Unsupported provider: {self.provider}")
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logging.warning(f"API call failed, retrying... Error: {str(e)}")
                time.sleep(1)

# Custom Memory Management
class ConversationMemory:
    def __init__(self, conversation_history: Optional[List[Tuple[str, str]]] = None):
        self.history = conversation_history or []
    
    def clear(self):
        """Clear the conversation history."""
        self.history = []
    
    def add_exchange(self, user_msg: str, ai_msg: str):
        """Add a user-AI exchange to the history."""
        self.history.append((user_msg, ai_msg))
    
    def get_formatted_history(self) -> str:
        """Get the formatted conversation history as a string."""
        return "\n".join(f"Human: {human}\nAI: {ai}" for human, ai in self.history)
    
    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Get the conversation history formatted for the OpenAI API."""
        messages = []
        for human, ai in self.history:
            messages.append({"role": "user", "content": human})
            messages.append({"role": "assistant", "content": ai})
        return messages

# Conversation Manager
class ConversationManager:
    def __init__(self, llm_client: LLMClient, memory: ConversationMemory, prompt_template: str):
        self.llm = llm_client
        self.memory = memory
        self.prompt_template = prompt_template
    
    def predict(self, user_input: str) -> str:
        """Generate a response for the user input."""
        # Format the prompt with conversation history
        formatted_prompt = self.prompt_template.format(
            conversation_history=self.memory.get_formatted_history(),
            input=user_input
        )
        
        # Prepare messages for API
        messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
        messages.extend(self.memory.get_messages_for_api())
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        response = self.llm.generate(messages)
        
        # Save to memory
        self.memory.add_exchange(user_input, response)
        
        return response

# Agent mode function definitions for OpenAI
AGENT_FUNCTIONS = [
    {
        "name": "send_email",
        "description": "Send an email to remind about a task or action item",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "Email address of the recipient. If not specified, defaults to jaime.raldua.veuthey@gmail.com"
                },
                "task_description": {
                    "type": "string",
                    "description": "Short description of what needs to be done"
                },
                "details": {
                    "type": "string",
                    "description": "Detailed information about the task"
                },
                "original_text": {
                    "type": "string",
                    "description": "The original text from the user's Telegram message"
                }
            },
            "required": ["task_description", "details", "original_text"]
        }
    },
    {
        "name": "create_notion_task",
        "description": "Create a task in Notion with a due date of tomorrow",
        "parameters": {
            "type": "object",
            "properties": {
                "task_title": {
                    "type": "string",
                    "description": "Title of the task (without the prefix, it will be added automatically)"
                },
                "task_details": {
                    "type": "string",
                    "description": "Detailed description of what needs to be done"
                },
                "original_text": {
                    "type": "string",
                    "description": "The original text from the user's Telegram message"
                }
            },
            "required": ["task_title", "task_details", "original_text"]
        }
    }
]

def save_journal_entry(entry, S3_BUCKET, journal_type):
    filename = JOURNAL_FILE if journal_type == "journal" else JOURNALGPT_FILE
    try:
        try:
            # Try to get the existing file
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=filename)
            data = json.loads(response['Body'].read().decode('utf-8'))
        except s3_client.exceptions.NoSuchKey:
            # If the file doesn't exist, create an empty list
            data = []

        timestamp = int(time.time())
        entry_data = {
            "timestamp": timestamp,
            "human_readable_date": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            "entry": entry
        }
        data.append(entry_data)

        # Save the updated data back to S3
        s3_client.put_object(Bucket=S3_BUCKET, Key=filename, Body=json.dumps(data))
        
        print(f"Journal entry saved successfully to {filename}")
    except Exception as e:
        logging.error(f"Error saving journal entry: {str(e)}")
        print(f"Error saving journal entry: {str(e)}")

def is_in_journalgpt_mode(username, S3_BUCKET):
    conversation_history = load_conversation_history(username, S3_BUCKET)
    for message, _ in reversed(conversation_history):
        if message.strip().lower() == "/journalgpt":
            return True
        elif message.strip().lower().startswith("/"):
            return False
    return False

def get_journalgpt_response(text, username, S3_BUCKET, OPENAI_API_KEY, GOOGLE_API_KEY, TELEGRAM_BOT_TOKEN, chat_id, model_name):
    journalgpt_prompt_template = """You are an AI coach assistant for journaling. Your role is to provide supportive, 
    insightful, and thought-provoking responses to the user's journal entries. Encourage self-reflection, personal growth, 
    and emotional awareness. Be empathetic and non-judgmental in your responses. If appropriate, you may ask follow-up 
    questions to help the user explore their thoughts and feelings more deeply.

    Here is the conversation history:
    {conversation_history}

    Human: {input}
    AI Coach:"""

    conversation_history = load_conversation_history(username, S3_BUCKET)
    journalgpt_history = []
    for message, response in reversed(conversation_history):
        if message.strip().lower() == "/journalgpt":
            break
        journalgpt_history.insert(0, (message, response))

    # Create LLM client, memory, and conversation manager based on MODEL_PROVIDER
    if MODEL_PROVIDER == "gemini":
        llm_client = LLMClient(provider="gemini", api_key=GOOGLE_API_KEY, model_name=model_name)
    else:
        llm_client = LLMClient(provider="openai", api_key=OPENAI_API_KEY, model_name=model_name)
    
    memory = ConversationMemory(conversation_history=journalgpt_history)
    conversation = ConversationManager(llm_client, memory, journalgpt_prompt_template)

    return handle_context_length(text, conversation, TELEGRAM_BOT_TOKEN, chat_id, model_name)


def is_in_journal_mode(username, S3_BUCKET):
    conversation_history = load_conversation_history(username, S3_BUCKET)
    for message, _ in reversed(conversation_history):
        if message.strip().lower() == "/journal":
            return True
        elif message.strip().lower().startswith("/"):
            return False
    return False

def is_in_agent_mode(username, S3_BUCKET):
    conversation_history = load_conversation_history(username, S3_BUCKET)
    for message, _ in reversed(conversation_history):
        if message.strip().lower() == "/agent":
            return True
        elif message.strip().lower().startswith("/"):
            return False
    return False

def process_generic_link(url, max_attempts=5, initial_timeout=5, transcript_only=False):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # Process arXiv URLs
    if 'arxiv.org' in url:
        if '/abs/' in url:
            url = url.replace('/abs/', '/pdf/')
        if not url.endswith('.pdf'):
            url += '.pdf'

    # Add Jina AI prefix
    jina_url = f"https://r.jina.ai/{url}"

    # Set up retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    timeout = initial_timeout
    for attempt in range(max_attempts):
        try:
            response = session.get(jina_url, headers=headers, timeout=timeout)
            response.raise_for_status()
            break  # If successful, break out of the loop
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout error for {jina_url} with timeout {timeout}s. Retrying...")
            timeout += 5  # Increase timeout by 5 seconds for each attempt
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching URL {jina_url}: {str(e)}")
            
            # Try direct URL if Jina AI fails
            try:
                response = session.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                break  # If successful, break out of the loop
            except requests.exceptions.RequestException as e:
                if attempt == max_attempts - 1:  # If this is the last attempt
                    return f"Couldn't retrieve content from {url}. Error: {str(e)}"
                logging.warning(f"Error fetching direct URL {url}: {str(e)}. Retrying...")
                timeout += 5  # Increase timeout for direct URL attempts as well
    else:
        return f"Couldn't retrieve content from {url} after {max_attempts} attempts."

    # If it's a PDF (likely for arXiv papers), return a message
    if response.headers.get('Content-Type', '').lower() == 'application/pdf':
        return f"This is a PDF document from {url}. PDF content extraction is not supported in this function."

    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text content
    text = soup.get_text()
    
    # Break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # Break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # Drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    if "New Comment Submit" in text: #so that the comments are not included
        text = text[:text.index("New Comment Submit")] 

    if transcript_only and len(text) > 4000:
        return jina_url
    return text

def process_youtube_link(parsed_url, RAPIDAPI_KEY="", transcript_only=False):
    if 'youtube.com' in parsed_url.netloc:
        query = parse_qs(parsed_url.query)
        video_id = query.get('v', [None])[0]
    elif 'youtu.be' in parsed_url.netloc:
        video_id = parsed_url.path.lstrip('/')
    else:
        return "Not a valid YouTube URL"
    
    if not video_id:
        return "Couldn't extract video ID"
    
    try:
        # Attempt to retrieve the transcript using YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript])
        
        if transcript_only and len(transcript_text) > 4000:
            return f"<showing only the first 4000 characters from '{len(transcript_text)}>: " + transcript_text[:4000]
        print(f"DEBUG Success fetching transcript using YouTubeTranscriptApi: {transcript_text[:100]}...")
        return transcript_text

    except Exception as e:
        print(f"DEBUG error using YouTubeTranscriptApi: {str(e)}")
        
        # If it fails, try the alternative method using RapidAPI
        url = "https://youtube-transcripts.p.rapidapi.com/youtube/transcript"
        querystring = {"url": f"https://youtu.be/{video_id}", "chunkSize": "500"}
        headers = {
            "x-rapidapi-key": RAPIDAPI_KEY,
            "x-rapidapi-host": "youtube-transcripts.p.rapidapi.com"
        }
        
        try:
            response = requests.get(url, headers=headers, params=querystring)
            response.raise_for_status()  # Raise an error for bad responses
            
            # Check if the response contains transcript data
            if response.json():
                transcript_data = response.json()
                transcript_text = " ".join([entry['text'] for entry in transcript_data['content']])
                
                if transcript_only and len(transcript_text) > 4000:
                    return f"<showing only the first 4000 characters from '{len(transcript_text)}>: " + transcript_text[:4000]
                print(f"DEBUG Success fetching transcript from RapidAPI: {transcript_text[:100]}...")
                return transcript_text
            
        except Exception as e:
            print(f"DEBUG error fetching transcript from RapidAPI: {str(e)}")
            return "Couldn't retrieve transcript using both methods."
        

def get_previous_user_message(username, S3_BUCKET):
    conversation_history = load_conversation_history(username, S3_BUCKET)
    for message, _ in reversed(conversation_history):
        if not message.strip().lower().startswith("/"):
            return message
    return None

def ask_stampy(question):
    url = "https://chat.stampy.ai:8443/chat"
    
    payload = {
        "stream": False,
        "query": question
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying Stampy: {str(e)}")
        return None
    
def format_text_response(text):
    lines = text.split('\n')
    formatted_text = ""
    current_indent = 0
    for line in lines:
        stripped_line = line.strip()
        if any(stripped_line.startswith(f"{i}.") for i in range(1, 10)):
            current_indent = 0
            formatted_text += '\n' + line + '\n'
        elif any(stripped_line.startswith(f"{c})") for c in 'abcdefghijklmnopqrstuvwxyz'):
            current_indent = 3
            formatted_text += ' ' * current_indent + line + '\n'
        else:
            if stripped_line:
                formatted_text += ' ' * current_indent + line + '\n'
    return formatted_text.strip()

def generate_uuid(timestamp, text):
    hash_object = hashlib.sha256(f"{timestamp}-{text}".encode('utf-8'))
    return hash_object.hexdigest()

def get_parameter(ssm, name):
    """Retrieve a parameter by name from AWS Systems Manager Parameter Store or from environment."""
    if is_running_in_lambda():
        print(f"Getting parameter '{name}' from the AWS Parameter Store.")
        response = ssm.get_parameter(Name=name, WithDecryption=True)
        return response['Parameter']['Value']
    else:
        # For local environment, get from environment variables
        env_key = name.split('/')[-1]  # Extract the last part of the path
        value = os.environ.get(env_key)
        if value:
            logger.info(f"Got parameter '{env_key}' from environment variable")
            return value
        else:
            # If not in environment, try AWS Parameter Store anyway
            if ssm:
                logger.info(f"Getting parameter '{name}' from AWS Parameter Store (local fallback)")
                response = ssm.get_parameter(Name=name, WithDecryption=True)
                return response['Parameter']['Value']
            else:
                raise ValueError(f"Parameter '{env_key}' not found in environment variables")

def send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, text):
    """Send a message to the bot, splitting if necessary to handle Telegram's 4096 char limit."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    
    # Split long messages
    max_length = 4096
    if len(text) <= max_length:
        data = {"chat_id": chat_id, "text": text}
        try:
            response = requests.post(url, data=data)
            response.raise_for_status()
            result = response.json()
            if not result.get('ok'):
                logger.error(f"Telegram API error: {result.get('description', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error sending message to Telegram: {str(e)}")
    else:
        # Split the message into chunks
        chunks = []
        while text:
            # Find a good breaking point (newline or space) before max_length
            if len(text) <= max_length:
                chunks.append(text)
                break
            
            # Look for newline first, then space
            break_point = text.rfind('\n', 0, max_length)
            if break_point == -1:
                break_point = text.rfind(' ', 0, max_length)
            if break_point == -1:
                break_point = max_length
            
            chunks.append(text[:break_point])
            text = text[break_point:].lstrip()
        
        # Send each chunk
        for i, chunk in enumerate(chunks):
            if i > 0:
                time.sleep(0.5)  # Rate limiting between messages
            
            data = {"chat_id": chat_id, "text": chunk}
            try:
                response = requests.post(url, data=data)
                response.raise_for_status()
                result = response.json()
                if not result.get('ok'):
                    logger.error(f"Telegram API error on chunk {i+1}/{len(chunks)}: {result.get('description', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Error sending message chunk {i+1}/{len(chunks)} to Telegram: {str(e)}")

def add_mention_to_agent_mentions_task(task_description, details, original_text, notion_token):
    """Add a mention to JaimeRV in the Agent Mentions task."""
    from datetime import datetime
    
    # Initialize Notion client
    notion = NotionClient(auth=notion_token)
    
    # The specific Agent Mentions task page ID from the URL
    agent_mentions_page_id = "23cfcfd1de9d80b58251ec5aa9c74d24"
    
    try:
        # Create blocks for the new mention section
        mention_blocks = [
            {
                "object": "block",
                "type": "divider",
                "divider": {}
            },
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [
                        {
                            "text": {
                                "content": f"Task from Telegram Agent - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                            }
                        }
                    ]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "mention",
                            "mention": {
                                "type": "user",
                                "user": {"id": NOTION_JAIME_USER_ID}
                            }
                        },
                        {
                            "type": "text",
                            "text": {
                                "content": " - New task reminder from Telegram Agent:"
                            }
                        }
                    ]
                }
            },
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [
                        {
                            "text": {
                                "content": "Task Description"
                            }
                        }
                    ]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "text": {
                                "content": f"{task_description}\n\n{details}"
                            }
                        }
                    ]
                }
            },
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [
                        {
                            "text": {
                                "content": "Original Telegram Message"
                            }
                        }
                    ]
                }
            },
            {
                "object": "block",
                "type": "quote",
                "quote": {
                    "rich_text": [
                        {
                            "text": {
                                "content": original_text
                            }
                        }
                    ]
                }
            }
        ]
        
        # Append the blocks to the Agent Mentions page
        notion.blocks.children.append(
            block_id=agent_mentions_page_id,
            children=mention_blocks
        )
        
        return True, "Mention added to Agent Mentions task"
    except Exception as e:
        logger.error(f"Failed to add mention to Agent Mentions task: {str(e)}")
        return False, str(e)

def handle_agent_send_email(args, mailgun_api_key, mailgun_domain, notion_token):
    """Handle email sending for agent mode and add Notion mention."""
    recipient = args.get("recipient", "jaime.raldua.veuthey@gmail.com")
    task_description = args.get("task_description", "Task from Telegram Agent")
    details = args.get("details", "")
    original_text = args.get("original_text", "")
    
    # First, add mention to Notion
    notion_success, notion_result = add_mention_to_agent_mentions_task(
        task_description, details, original_text, notion_token
    )
    
    # Then send email
    subject = f"[Telegram Agent] {task_description}"
    
    html_content = f"""
    <html>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
        <h2>Task Reminder from Telegram Agent</h2>
        <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h3>Task Description:</h3>
        <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
            <p><strong>{task_description}</strong></p>
            <p>{details}</p>
        </div>
        
        <h3>Original Telegram Message:</h3>
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; font-style: italic;">
            <p>{original_text}</p>
        </div>
        
        <p style="margin-top: 20px; font-size: 12px; color: #666;">
            This email was automatically generated by the Telegram AI Agent
        </p>
    </body>
    </html>
    """
    
    try:
        response = requests.post(
            f"https://api.mailgun.net/v3/{mailgun_domain}/messages",
            auth=("api", mailgun_api_key),
            data={
                "from": f"Telegram Agent <mailgun@{mailgun_domain}>",
                "to": [recipient],
                "subject": subject,
                "html": html_content
            }
        )
        
        email_success = response.status_code == 200
        if email_success:
            result = response.json()
            email_msg = f"Email sent to {recipient} (ID: {result.get('id', 'Success')})"
        else:
            email_msg = f"Failed to send email: {response.status_code} - {response.text}"
        
        # Combine results
        combined_results = []
        if email_success:
            combined_results.append(email_msg)
        else:
            combined_results.append(f"Email error: {email_msg}")
            
        if notion_success:
            combined_results.append("Notion mention added")
        else:
            combined_results.append(f"Notion error: {notion_result}")
        
        # Return overall success only if both succeeded
        overall_success = email_success and notion_success
        return overall_success, " | ".join(combined_results)
        
    except Exception as e:
        logger.error(f"Failed to send agent email: {str(e)}")
        return False, f"Email error: {str(e)} | Notion: {'Success' if notion_success else notion_result}"

def handle_agent_create_notion_task(args, notion_token):
    """Handle Notion task creation for agent mode."""
    from datetime import datetime, timedelta
    
    task_title = args.get("task_title", "Task")
    task_details = args.get("task_details", "")
    original_text = args.get("original_text", "")
    
    # Initialize Notion client
    notion = NotionClient(auth=notion_token)
    
    # Calculate tomorrow's date
    tomorrow = (datetime.now() + timedelta(days=1)).date().isoformat()
    
    # Create the task properties
    properties = {
        "Task name": {
            "title": [
                {
                    "text": {
                        "content": f"(Telegram Jaime Agent) {task_title}"
                    }
                }
            ]
        },
        "Due Date": {
            "date": {
                "start": tomorrow
            }
        },
        "Assign": {
            "people": [
                {
                    "id": NOTION_JAIME_USER_ID
                }
            ]
        }
    }
    
    try:
        # Create the task page
        page = notion.pages.create(
            parent={"database_id": NOTION_TASKS_DATABASE_ID},
            properties=properties
        )
        
        # Add comment with mention to JaimeRV
        comment_blocks = [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "mention",
                            "mention": {
                                "type": "user",
                                "user": {"id": NOTION_JAIME_USER_ID}
                            }
                        },
                        {
                            "type": "text",
                            "text": {
                                "content": " This task was created from the Telegram Jaime Agent bot."
                            }
                        }
                    ]
                }
            },
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [
                        {
                            "text": {
                                "content": "Task Details"
                            }
                        }
                    ]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "text": {
                                "content": task_details
                            }
                        }
                    ]
                }
            },
            {
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [
                        {
                            "text": {
                                "content": "Original Telegram Message"
                            }
                        }
                    ]
                }
            },
            {
                "object": "block",
                "type": "quote",
                "quote": {
                    "rich_text": [
                        {
                            "text": {
                                "content": original_text
                            }
                        }
                    ]
                }
            }
        ]
        
        # Add the comment blocks to the page
        notion.blocks.children.append(
            block_id=page["id"],
            children=comment_blocks
        )
        
        # Generate the Notion page URL
        page_id = page["id"].replace("-", "")
        notion_url = f"https://www.notion.so/{page_id}"
        
        return True, f"Task created in Notion: '{task_title}' (due {tomorrow})\n🔗 {notion_url}"
    except Exception as e:
        logger.error(f"Failed to create Notion task: {str(e)}")
        return False, f"Failed to create task: {str(e)}"

def send_email_via_mailgun(recipient_email, subject, conversation_data, additional_info="", mailgun_api_key=None, mailgun_domain=None):
    """Send email using Mailgun API with conversation summary."""
    try:
        # Format the conversation history
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <h2>Telegram Bot Conversation</h2>
            <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Username:</strong> {conversation_data.get('username', 'Unknown')}</p>
            
            {f'<p><strong>Additional Info:</strong> {additional_info}</p>' if additional_info else ''}
            
            <h3>Conversation History:</h3>
            <div style="background-color: #f5f5f5; padding: 15px; border-radius: 5px;">
        """
        
        # Add conversation history
        conversation_history = conversation_data.get('history', [])
        if conversation_history:
            for msg, response in conversation_history:
                # Escape HTML to prevent injection
                msg = msg.replace('<', '&lt;').replace('>', '&gt;')
                response = response.replace('<', '&lt;').replace('>', '&gt;')
                html_content += f"""
                <div style="margin-bottom: 15px;">
                    <p style="color: #0066cc;"><strong>User:</strong> {msg}</p>
                    <p style="color: #333;"><strong>Bot:</strong> {response}</p>
                </div>
                <hr style="border: none; border-top: 1px solid #ddd;">
                """
        else:
            html_content += "<p>No conversation history available.</p>"
        
        html_content += """
            </div>
            <p style="margin-top: 20px; font-size: 12px; color: #666;">
                This email was sent from the AIS Digest Telegram Bot
            </p>
        </body>
        </html>
        """
        
        # Send email via Mailgun
        response = requests.post(
            f"https://api.mailgun.net/v3/{mailgun_domain}/messages",
            auth=("api", mailgun_api_key),
            data={
                "from": f"AIS Digest Bot <mailgun@{mailgun_domain}>",
                "to": [recipient_email],
                "subject": subject,
                "html": html_content
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Email sent successfully via Mailgun: {result.get('id', 'No ID')}")
            return True, result.get('id', 'Success')
        else:
            error_msg = f"Mailgun API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return False, error_msg
        
    except Exception as e:
        logger.error(f"Failed to send email via Mailgun: {str(e)}")
        return False, str(e)

def send_audio_to_bot(TELEGRAM_BOT_TOKEN, chat_id, text):
    try:
        response = polly.synthesize_speech(
            OutputFormat='mp3',
            Text=text,
            VoiceId=voice_name
        )
        audio = None
        if "AudioStream" in response:
            with closing(response["AudioStream"]) as stream:
                audio = stream.read()

        if audio:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendAudio"
            data = {"chat_id": chat_id}
            files = {"audio": io.BytesIO(audio)}
            requests.post(url, data=data, files=files)
        else:
            logging.error("No audio stream found in Polly response")
    except Exception as e:
        logging.error(f"Error sending audio: {str(e)}")
        # Optionally, send a message to the user about the audio error
        send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, "<Sorry, there was an error sending the audio version of the message.>")


def get_encoding(model_name: str):
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        print(f"Warning: Model '{model_name}' not found. Using cl100k_base encoding.")
        return tiktoken.get_encoding("cl100k_base")

def num_tokens_from_messages(messages: List[Union[Dict[str, str], Tuple[str, str]]], model_name: str) -> int:
    encoding = get_encoding(model_name)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        if isinstance(message, dict):
            # API format message
            role = message.get("role", "user")
            content = message.get("content", "")
            num_tokens += len(encoding.encode(content))
        elif isinstance(message, tuple):
            # Conversation history format (user_message, ai_response)
            user_message, ai_response = message
            num_tokens += len(encoding.encode(user_message))
            num_tokens += len(encoding.encode(ai_response))
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def trim_messages(
    messages: List[Dict[str, str]],
    max_tokens: int,
    token_counter: Callable[[List[Dict[str, str]]], int],
    strategy: str = "last",
    include_system: bool = True,
) -> List[Dict[str, str]]:
    if strategy not in ["last", "first"]:
        raise ValueError(f"Invalid strategy: {strategy}")
    
    system_message = next((m for m in messages if m.get("role") == "system"), None)
    non_system_messages = [m for m in messages if m.get("role") != "system"]
    
    if include_system and system_message:
        max_tokens -= token_counter([system_message])
    
    while token_counter(non_system_messages) > max_tokens:
        if strategy == "last":
            non_system_messages.pop(0)
        else:
            non_system_messages.pop()
    
    if include_system and system_message:
        return [system_message] + non_system_messages
    return non_system_messages

def split_context(context: str, model_name: str, max_tokens: int = MAX_TOKENS) -> List[str]:
    """Splits the context into chunks that are smaller than the max token limit."""
    encoding = get_encoding(model_name)
    tokens = encoding.encode(context)
    chunks = []
    current_chunk = []

    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= max_tokens - 1000:  # Leave some room for the response
            chunks.append(encoding.decode(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(encoding.decode(current_chunk))

    return chunks

def process_chunks(chunks: List[str], conversation: ConversationManager, TELEGRAM_BOT_TOKEN: str, chat_id: int) -> str:
    """Processes each chunk and returns the final response."""
    responses = []
    for i, chunk in enumerate(chunks):
        response = conversation.predict(f"Chunk {i+1}/{len(chunks)}: {chunk}")
        responses.append(response)
        
        # Send a progress update to the user
        progress_message = f"Processing chunk {i+1} of {len(chunks)}..."
        send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, progress_message)

    final_response = "\n\n".join(responses)
    return final_response

def handle_context_length(text: str, conversation: ConversationManager, TELEGRAM_BOT_TOKEN: str, chat_id: int, model_name: str) -> str:
    try:
        return conversation.predict(text)
    except Exception as e:
        if "maximum context length" in str(e).lower():
            warning_message = ("The context was too long. Trimming conversation history to fit within the token limit.")
            send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, warning_message)
            
            # Convert conversation history to API format
            messages = [{"role": "system", "content": "You are a helpful AI assistant."}]
            messages.extend(conversation.memory.get_messages_for_api())
            messages.append({"role": "user", "content": text})
            
            trimmed_messages = trim_messages(
                messages,
                max_tokens=MAX_TOKENS - 1000,  # Leave some room for the response
                token_counter=lambda msgs: num_tokens_from_messages(msgs, model_name),
                strategy="last",
                include_system=True
            )
            
            # Rebuild conversation history from trimmed messages
            new_history = []
            i = 1  # Skip system message
            while i < len(trimmed_messages) - 1:  # Exclude the last message (current input)
                if trimmed_messages[i]["role"] == "user" and i + 1 < len(trimmed_messages) and trimmed_messages[i + 1]["role"] == "assistant":
                    new_history.append((trimmed_messages[i]["content"], trimmed_messages[i + 1]["content"]))
                    i += 2
                else:
                    i += 1
            
            conversation.memory.history = new_history
            
            return conversation.predict(text)
        elif "You exceeded your current quota, please check your plan and billing details." in str(e):
            return "Sorry, I'm currently out of credits. Please try again later."
        else:
            raise e

def handle_agent_mode(text: str, username: str, S3_BUCKET: str, OPENAI_API_KEY: str, TELEGRAM_BOT_TOKEN: str, chat_id: int, MAILGUN_API_KEY: str, MAILGUN_DOMAIN: str, NOTION_TOKEN: str) -> str:
    """Handle agent mode interactions with function calling."""
    # Prepare the agent prompt
    agent_prompt = f"""You are an AI agent that helps users automate tasks. Based on the user's request, determine which function to call.
    
Available functions:
1. send_email - Send an email reminder about a task
2. create_notion_task - Create a task in Notion with tomorrow as the due date

Analyze the user's request and call the appropriate function. If the request mentions both email and Notion, prioritize based on what seems to be the primary intent.

User request: {text}"""
    
    # Create messages for OpenAI
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant that can send emails and create Notion tasks."},
        {"role": "user", "content": agent_prompt}
    ]
    
    try:
        # Always use OpenAI for agent mode (function calling)
        # Use the configured OpenAI model (e.g., o4-mini) from the environment
        agent_model = OPENAI_MODEL  # Uses the OPENAI_MODEL from config
        llm_client = LLMClient(provider="openai", api_key=OPENAI_API_KEY, model_name=agent_model)
        
        # Generate response with function calling
        response = llm_client.generate(messages, functions=AGENT_FUNCTIONS)
        
        if isinstance(response, dict) and response.get("type") == "function_call":
            function_call = response["function_call"]
            function_name = function_call.name
            function_args = json.loads(function_call.arguments)
            
            # Add original text to args if not present
            if "original_text" not in function_args:
                function_args["original_text"] = text
            
            # Execute the appropriate function
            if function_name == "send_email":
                success, result = handle_agent_send_email(function_args, MAILGUN_API_KEY, MAILGUN_DOMAIN, NOTION_TOKEN)
                if success:
                    return f"✅ {result}"
                else:
                    return f"❌ Error: {result}"
            
            elif function_name == "create_notion_task":
                success, result = handle_agent_create_notion_task(function_args, NOTION_TOKEN)
                if success:
                    return f"✅ {result}"
                else:
                    return f"❌ Error: {result}"
            
            else:
                return f"Unknown function: {function_name}"
        
        else:
            # If no function was called, return the AI's response
            return f"I understood your request but couldn't determine the appropriate action. Response: {response}"
    
    except Exception as e:
        logger.error(f"Error in agent mode: {str(e)}")
        return f"❌ Error processing agent command: {str(e)}"

def generate_response(text: str, username: str, S3_BUCKET: str, OPENAI_API_KEY: str, GOOGLE_API_KEY: str, TELEGRAM_BOT_TOKEN: str, chat_id: int, model_name: str) -> str:
    conversation_history = load_conversation_history(username, S3_BUCKET)
    
    # Create LLM client, memory, and conversation manager based on MODEL_PROVIDER
    if MODEL_PROVIDER == "gemini":
        llm_client = LLMClient(provider="gemini", api_key=GOOGLE_API_KEY, model_name=model_name)
    else:
        llm_client = LLMClient(provider="openai", api_key=OPENAI_API_KEY, model_name=model_name)
    
    memory = ConversationMemory(conversation_history=conversation_history)
    conversation = ConversationManager(llm_client, memory, prompt_template)

    return handle_context_length(text, conversation, TELEGRAM_BOT_TOKEN, chat_id, model_name)

def load_conversation_history(username, S3_BUCKET):
    filename = f"userlogs/{username}.json"
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=filename)
        data = json.loads(response['Body'].read().decode('utf-8'))
        full_history = [(item['text'], item['response']) for item in data]
        
        last_new_index = -1
        for i, (text, _) in enumerate(reversed(full_history)):
            if "/new" in text:
                last_new_index = len(full_history) - i - 1
                break
        
        return full_history[last_new_index + 1:] if last_new_index != -1 else full_history
    except Exception as e:
        print(f"WARNING: in load_conversation_history() there was an error: '{e}'")
        return []

def save_message_to_json(username, message_data, S3_BUCKET):
    filename = f"userlogs/{username}.json"
    try:
        s3_client.head_bucket(Bucket=S3_BUCKET)
    except s3_client.exceptions.NoSuchBucket:
        s3_client.create_bucket(Bucket=S3_BUCKET)

    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=filename)
        data = json.loads(response['Body'].read().decode('utf-8'))
    except s3_client.exceptions.NoSuchKey:
        data = []

    uuid = message_data["uuid"]
    existing_uuids = [msg["uuid"] for msg in data]
    if uuid not in existing_uuids:
        data.append(message_data)
        s3_client.put_object(Bucket=S3_BUCKET, Key=filename, Body=json.dumps(data))
    else:
        logging.warning(f"Message with uuid {uuid} already exists. It will not be written in the userlog.")

def get_user_chat_id(user_id):
    s3_client = boto3.client('s3')
    bucket_name = 'ais-digest'
    file_key = f'userlogs/{user_id}.json'

    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = response['Body'].read().decode('utf-8')
        user_data = json.loads(file_content)
        
        if user_data and len(user_data) > 0:
            first_message = user_data[0]
            chat_id = first_message.get('chat', {}).get('id')
            
            if chat_id:
                return chat_id
            else:
                print(f"Chat ID not found for user {user_id}")
                return None
        else:
            print(f"No messages found for user {user_id}")
            return None
    
    except s3_client.exceptions.NoSuchKey:
        print(f"No JSON file found for user {user_id}")
        return None
    except Exception as e:
        print(f"Error retrieving chat ID for user {user_id}: {str(e)}")
        return None

def get_previous_url(username, S3_BUCKET):
    conversation_history = load_conversation_history(username, S3_BUCKET)
    for message, _ in reversed(conversation_history):
        if message.strip().lower().startswith("http") or message.strip().lower().startswith("www"):
            return message.strip().split('\n')[0]  # Return the message till the first newline
    return None

def send_document_to_bot(TELEGRAM_BOT_TOKEN, chat_id, file_content, filename, caption=""):
    """Send a document to the bot via Telegram's sendDocument API."""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption
        
        files = {"document": (filename, io.BytesIO(file_content.encode('utf-8')), 'text/markdown')}
        response = requests.post(url, data=data, files=files)
        response.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Error sending document: {str(e)}")
        return False

def load_content_processor_state():
    """Load the state file from the content processor S3 bucket."""
    try:
        response = s3_client.get_object(Bucket=CONTENT_PROCESSOR_BUCKET, Key=CONTENT_PROCESSOR_STATE_FILE)
        data = json.loads(response['Body'].read().decode('utf-8'))
        return data
    except s3_client.exceptions.NoSuchKey:
        logging.warning(f"State file not found: {CONTENT_PROCESSOR_STATE_FILE}")
        return {"youtube": {}, "readwise": {}}
    except Exception as e:
        logging.error(f"Error loading state file: {str(e)}")
        return {"youtube": {}, "readwise": {}}

def save_content_processor_state(state_data):
    """Save the updated state file to the content processor S3 bucket."""
    try:
        s3_client.put_object(
            Bucket=CONTENT_PROCESSOR_BUCKET, 
            Key=CONTENT_PROCESSOR_STATE_FILE, 
            Body=json.dumps(state_data, indent=2)
        )
        return True
    except Exception as e:
        logging.error(f"Error saving state file: {str(e)}")
        return False

def get_content_file(file_path):
    """Download content file from S3."""
    try:
        response = s3_client.get_object(Bucket=CONTENT_PROCESSOR_BUCKET, Key=file_path)
        return response['Body'].read().decode('utf-8')
    except Exception as e:
        logging.error(f"Error downloading content file {file_path}: {str(e)}")
        return None

def handle_retrieve_command(TELEGRAM_BOT_TOKEN, chat_id):
    """Handle the /retrieve command to fetch and deliver unprocessed content."""
    try:
        # Load state file
        state_data = load_content_processor_state()
        
        youtube_count = 0
        readwise_count = 0
        total_delivered = 0
        
        # Process YouTube content
        if "youtube" in state_data:
            for video_id, item_data in state_data["youtube"].items():
                if item_data.get("processed", False) and not item_data.get("retrieved", False):
                    # Download the content file
                    file_content = get_content_file(item_data["file_path"])
                    if file_content:
                        # Send as document
                        filename = f"youtube_{video_id}.md"
                        caption = f"📄 YouTube: {item_data['title']}\n🔗 Original: {item_data['url']}"
                        
                        if send_document_to_bot(TELEGRAM_BOT_TOKEN, chat_id, file_content, filename, caption):
                            # Mark as retrieved
                            item_data["retrieved"] = True
                            item_data["retrieved_date"] = datetime.utcnow().isoformat() + "Z"
                            youtube_count += 1
                            total_delivered += 1
                            
                            # Rate limiting: 1 second delay between messages
                            time.sleep(1)
                        else:
                            logging.error(f"Failed to send YouTube video: {video_id}")
        
        # Process Readwise content
        if "readwise" in state_data:
            for article_id, item_data in state_data["readwise"].items():
                if item_data.get("processed", False) and not item_data.get("retrieved", False):
                    # Download the content file
                    file_content = get_content_file(item_data["file_path"])
                    if file_content:
                        # Send as document
                        filename = f"readwise_{article_id}.md"
                        caption = f"📄 Readwise: {item_data['title']}\n🔗 Original: {item_data['url']}"
                        
                        if send_document_to_bot(TELEGRAM_BOT_TOKEN, chat_id, file_content, filename, caption):
                            # Mark as retrieved
                            item_data["retrieved"] = True
                            item_data["retrieved_date"] = datetime.utcnow().isoformat() + "Z"
                            readwise_count += 1
                            total_delivered += 1
                            
                            # Rate limiting: 1 second delay between messages
                            time.sleep(1)
                        else:
                            logging.error(f"Failed to send Readwise article: {article_id}")
        
        # Save updated state
        if total_delivered > 0:
            if save_content_processor_state(state_data):
                summary = f"Retrieved {youtube_count} YouTube videos and {readwise_count} Readwise articles"
            else:
                summary = f"Delivered {youtube_count} YouTube videos and {readwise_count} Readwise articles, but failed to update state"
        else:
            summary = "No new content to retrieve"
        
        return summary
        
    except Exception as e:
        error_msg = f"Error processing retrieve command: {str(e)}"
        logging.error(error_msg)
        return error_msg

def lambda_handler(event, context):
    try:
        # Load model configuration from Parameter Store first
        try:
            model_provider = get_parameter(ssm, MODEL_PROVIDER_SSM_PATH)
            gemini_model = get_parameter(ssm, GEMINI_MODEL_SSM_PATH)
            openai_model = get_parameter(ssm, OPENAI_MODEL_SSM_PATH)
        except Exception as e:
            logger.warning(f"Failed to load model config from Parameter Store: {e}. Using defaults.")
            model_provider = DEFAULT_MODEL_PROVIDER
            gemini_model = DEFAULT_GEMINI_MODEL
            openai_model = DEFAULT_OPENAI_MODEL
        
        # Initialize model configuration
        initialize_model_config(model_provider, gemini_model, openai_model)
        MODEL_NAME = DEFAULT_MODEL

        # Load tokens from Parameter Store
        TELEGRAM_BOT_TOKEN = get_parameter(ssm, TELEGRAM_BOT_TOKEN_SSM_PATH)
        OPENAI_API_KEY = get_parameter(ssm, OPENAI_API_KEY_SSM_PATH)
        GOOGLE_API_KEY = get_parameter(ssm, GOOGLE_API_KEY_SSM_PATH)
        S3_BUCKET = get_parameter(ssm, S3_BUCKET_SSM_PATH)
        USERS_ALLOWED = get_parameter(ssm, USERS_ALLOWED_SSM_PATH).split(',')
        AUX_USERNAME = get_parameter(ssm, AUX_USERNAME_SSM_PATH)
        RAPIDAPI_KEY = get_parameter(ssm, RAPIDAPI_KEY_SSM_PATH)
        MAILGUN_API_KEY = get_parameter(ssm, MAILGUN_API_KEY_SSM_PATH)
        MAILGUN_DOMAIN = get_parameter(ssm, MAILGUN_DOMAIN_SSM_PATH)
        NOTION_TOKEN = get_parameter(ssm, NOTION_TOKEN_SSM_PATH)

        print(f"DEBUG: The event received is: {event}")
        message = json.loads(event['body'])['message']
        
        chat_id = message['chat'].get('id')
        print(f"DEBUG: The chat_id is '{chat_id}'")
        chat_type = message['chat'].get('type')
        print(f"DEBUG: The chat_type is '{chat_type}'")
        current_user = message['from'].get('username')
        print(f"DEBUG: The current_user is '{current_user}'")
        current_first_name = message['from'].get('first_name')
        print(f"DEBUG: The current_first_name is '{current_first_name}'")
        
        # Check if the message is from a private chat
        if chat_type != 'private':
            send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, f"Sorry, but I cannot work yet with chat type: '{chat_type}'. Please use me in a private chat.")
            return {
                'statusCode': 200,
                'body': json.dumps(f'Message sent: Bot not configured for {chat_type} chats.')
            }
        
        if not current_user and not current_first_name:
            error_message = "Error: Unable to identify the user. Please make sure you have a username set in Telegram or that your first name is set."
            send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, error_message)
            return {
                'statusCode': 400,
                'body': json.dumps('Error: Unable to identify the user.')
            }
        
        if current_first_name and not current_user:
            current_user = current_first_name.lower() #TODO: fix this because this is only a patch for the case when the user does not have a username set in Telegram but it has a first name
        
        if (current_user not in USERS_ALLOWED) and (current_user != AUX_USERNAME):
            send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, f"Sorry, but you first need to register to use this chatbot. Your current_user is {current_user}; your current_first_name is {current_first_name}; and the USERS_ALLOWED are {USERS_ALLOWED}.")
            return {
                'statusCode': 200,
                'body': json.dumps('User not registered.')
            }

        if 'voice' in message and message['voice']['mime_type'] == 'audio/ogg':
            api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
            file_id = message['voice']['file_id']
            response = requests.get(api_url + "/getFile", params={"file_id": file_id}).json()
            file_path = response["result"]["file_path"]
            
            audio_data = requests.get(f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}").content
            
            audio_file = io.BytesIO(audio_data)
            s3 = boto3.client('s3')
            s3.put_object(Bucket=S3_BUCKET, Key='temp_audio.ogg', Body=audio_file)

            TranscriptionJob = 'MyTranscriptionJob4Chatbot'+str(int(time.time()))
            transcribe = boto3.client('transcribe')
            transcribe.start_transcription_job(
                TranscriptionJobName=TranscriptionJob,
                Media={'MediaFileUri': f's3://{S3_BUCKET}/temp_audio.ogg'},
                MediaFormat='ogg',
                LanguageCode='en-US',
            )

            while True:
                status = transcribe.get_transcription_job(TranscriptionJobName=TranscriptionJob)
                if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                    break
                time.sleep(5)

            transcript_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            transcript_result = requests.get(transcript_url).json()
            text = transcript_result['results']['transcripts'][0]['transcript']    
            send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, f"My transcript of your audio is: {text}")

        else:
            text = message['text']

        timestamp = message['date']
        uuid = generate_uuid(timestamp, text)
        print(f"DEBUG: The text received is: {text}")

        # Check for any link starting with "https" in the text
        https_links = [word for word in text.split() if word.startswith("https://")]
        if https_links:
            # If there are https links, keep only the first one
            text = https_links[0]
        else:
            # If no https links are found, proceed as normal
            pass
        
        if text.strip().lower() == "/journal":
            response = "Journal mode activated. Your subsequent messages will be saved as journal entries. Use /new to exit journal mode."
        elif is_in_journal_mode(current_user, S3_BUCKET):
            if text.strip().lower() == "/new":
                response = "Journal mode deactivated. Your messages will now be processed normally."
            else:
                save_journal_entry(text, S3_BUCKET, "journal")
                response = "Your journal entry has been saved."
        elif text.strip().lower() == "/journalgpt":
            response = "JournalGPT mode activated. You're now in a conversation with an AI coach journaling assistant. Use /new to exit JournalGPT mode."
        elif is_in_journalgpt_mode(current_user, S3_BUCKET):
            if text.strip().lower() == "/new":
                response = "JournalGPT mode deactivated. Your messages will now be processed normally."
            else:
                response = get_journalgpt_response(text, current_user, S3_BUCKET, OPENAI_API_KEY, GOOGLE_API_KEY, TELEGRAM_BOT_TOKEN, chat_id, MODEL_NAME)
                save_journal_entry(f"User: {text}\nAI: {response}", S3_BUCKET, "journalgpt")
        elif text.strip().lower().startswith("/new"):
            text = "(/new) Let's start a new conversation."
            response = generate_response(text, current_user, S3_BUCKET, OPENAI_API_KEY, GOOGLE_API_KEY, TELEGRAM_BOT_TOKEN, chat_id, MODEL_NAME)
        elif text.strip().lower().startswith("/exam"):
            text = """(/exam) 
            Ask the user for his understanding on the article/transcript/podcast that is being discussed 

            You can ask him to summarize or explain you specific sections, or concepts. Then tell him where he is right or wrong and why. 
            You can challenge the knowledge of the user asking him to do any of these things:
            - Teach an imaginary student (role played by you the AI Assistant=
            - Draw a mindmap
            - Draw an image instead of using words (to find a visual way of expressing information)
            - Answer practice questions (created by the you the AI assistant)
            - Create your own challenging test questions
            - Create a test question that puts what you've learned into a real-world context
            - Take a difficult question you found in a practice test and modify it so that it involves different variables or adds an extra step
            - Form a study group (user + you the AI assistant) and quiz each other (for some subjects, you can even debate the topic, with one side trying to prove that the other person is missing a point or understanding it incorrectly)

            Do not make any information up and respond only with information from the text. If it does not appear or you do not know then say so without making up information.
            """
            response = generate_response(text, current_user, S3_BUCKET, OPENAI_API_KEY, GOOGLE_API_KEY, TELEGRAM_BOT_TOKEN, chat_id, MODEL_NAME)
        elif text.strip().lower().startswith("/reflect"):
            text = """(/reflect) 
            You are an AI assistant designed to function as a reflective journal for the user. Your primary role is to generate daily or periodic prompts that encourage the user to reflect deeply on various aspects of their life. These prompts should be designed to help the user gain insight into their thoughts, feelings, behaviors, and overall well-being. You will not receive or expect any responses from the user, so your goal is to provide thought-provoking, introspective prompts that the user can contemplate and perhaps write about elsewhere.

            Your prompts should cover a range of topics, including but not limited to these topics:

            - **Personal Feelings and Emotions:** Encourage the user to explore their current emotional state, underlying feelings, and emotional patterns over time. 
            - **Awareness of Inner States:** Help the user become aware of their internal experiences, such as physical sensations, thoughts, and emotional responses.
            - **Level of Energy:** Prompt the user to reflect on their physical and mental energy levels, identifying patterns, and possible causes.
            - **Level of Productivity:** Encourage the user to evaluate their productivity, identify obstacles, and consider ways to enhance their efficiency.
            - **Level of Focus:** Help the user assess their ability to concentrate, and explore factors that might affect their focus.

            You will select at the beginning of the conversation one of these topics randomly and will focus the conversation on that topic till the user says otherwise. So for example if you start talking about Levels of Energy do not change it to Feelings until the user says that he wants to change. The user can select the topic directly. 

            Use techniques from Cognitive Behavioral Therapy (CBT) to help the user challenge unhelpful thoughts and beliefs, recognize cognitive distortions, and develop healthier thought patterns. Additionally, incorporate elements from Internal Family Systems (IFS) to guide the user in exploring different parts of their psyche, such as the "inner child," "protector," or "manager," and facilitate communication and harmony between these parts.

            Sample prompts for the topics mentioned could be:

            1. **Emotional Reflection:** "Take a moment to reflect on how you're feeling today. What emotions are most prominent? What might these emotions be trying to tell you about your current situation or unmet needs?"
            
            2. **Cognitive Distortions:** "Think about a recent situation where you felt particularly stressed or upset. Are there any thoughts that might have been distorted or exaggerated? How could you reframe these thoughts to be more balanced?"

            3. **Energy Levels:** "Consider your energy levels throughout the day. When do you feel most energized, and when do you feel drained? What activities or thoughts seem to influence your energy?"

            4. **Productivity Reflection:** "Reflect on your productivity today. Were there any tasks that felt particularly difficult to start or complete? What might have been holding you back?"

            5. **Internal Dialogue:** "Consider a part of yourself that often criticizes or judges you. What might this part be trying to protect you from? How can you acknowledge its concerns while also challenging its harshness?"

            6. **Focus and Distractions:** "Think about a time today when you struggled to focus. What distractions were present? How might you minimize these distractions in the future?"

            Only when the user asks you to change the topic (saying "new" or "new topic") you will start again with a new topic. If not then with every prompt (e.g. the user say "more" or "in depth" or "next" you will help the user dive deeper in the topic that you have started without changing to a new one.

            Remember, your role is to provide supportive, non-judgmental prompts that encourage the user to think deeply and compassionately about themselves. Your prompts should be varied and cater to different aspects of the user's life and inner experiences, helping them to continuously grow and improve without needing to respond directly to you.
            """
            response = generate_response(text, current_user, S3_BUCKET, OPENAI_API_KEY, GOOGLE_API_KEY, TELEGRAM_BOT_TOKEN, chat_id, MODEL_NAME)
        elif text.strip().lower().startswith("/stampy"):
            stampy_query = text[7:].strip()
            
            if not stampy_query:
                stampy_query = get_previous_user_message(current_user, S3_BUCKET)
                if not stampy_query:
                    send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, "No previous question found to ask Stampy.")
                    return {
                        'statusCode': 200,
                        'body': json.dumps('No previous question found.')
                    }
            
            stampy_response = ask_stampy(stampy_query)
            if stampy_response:
                formatted_response = format_text_response(stampy_response)
                response = f"Stampy's response to '{stampy_query}':\n\n{formatted_response}"
            else:
                response = "Sorry, I couldn't get a response from Stampy."
        elif text.strip().lower().startswith("/transcript"):
            previous_url = get_previous_url(current_user, S3_BUCKET)
            if previous_url:
                parsed_url = urlparse(previous_url)
                if 'youtube' in parsed_url.netloc or 'youtu.be' in parsed_url.netloc:
                    content = process_youtube_link(parsed_url, RAPIDAPI_KEY,  transcript_only=True)
                    response = f"Here's the transcript of the video:\n\n{content}"
                else:
                    content = process_generic_link(previous_url, transcript_only=True)
                    response = f"Here's the transcript of the content:\n\n{content}"
            else:
                response = "No previous URL found. Please provide a YouTube URL for transcription."
            send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, response)
            return {
                'statusCode': 200,
                'body': json.dumps('Transcript request processed.')
            }
        elif text.strip().lower().startswith("/email"):
            # Extract any additional info after /email command
            email_parts = text.strip().split(maxsplit=1)
            additional_info = email_parts[1] if len(email_parts) > 1 else ""
            
            # Load conversation history
            conversation_history = load_conversation_history(current_user, S3_BUCKET)
            
            # Prepare conversation data
            conversation_data = {
                'username': current_user,
                'history': conversation_history
            }
            
            # Prepare email subject with unique timestamp
            subject = f"Telegram Bot Conversation - {current_user} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Send email
            success, result = send_email_via_mailgun(
                recipient_email="jaime.raldua.veuthey@gmail.com",
                subject=subject,
                conversation_data=conversation_data,
                additional_info=additional_info,
                mailgun_api_key=MAILGUN_API_KEY,
                mailgun_domain=MAILGUN_DOMAIN
            )
            
            if success:
                response = f"📧 Email sent successfully! Message ID: {result}"
            else:
                response = f"❌ Failed to send email: {result}"
        elif text.strip().lower().startswith("/retrieve"):
            send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, "🔄 Retrieving content from processor...")
            response = handle_retrieve_command(TELEGRAM_BOT_TOKEN, chat_id)
            send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, response)
            return {
                'statusCode': 200,
                'body': json.dumps('Retrieve request processed.')
            }
        elif text.strip().lower() == "/agent":
            response = "🤖 Agent mode activated! I can now help you:\n\n• Send email reminders\n• Create tasks in Notion\n\nJust tell me what you need to do, and I'll handle it for you.\n\nExamples:\n- 'Send me an email to remind me to do the dishes before tomorrow'\n- 'Create a Notion task to review the quarterly report'\n\nUse /new to exit agent mode."
        elif is_in_agent_mode(current_user, S3_BUCKET):
            if text.strip().lower() == "/new":
                response = "Agent mode deactivated. Your messages will now be processed normally."
            else:
                response = handle_agent_mode(text, current_user, S3_BUCKET, OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, chat_id, MAILGUN_API_KEY, MAILGUN_DOMAIN, NOTION_TOKEN)
        elif text.strip().lower().startswith("http") or text.strip().lower().startswith("www"):
            parsed_url = urlparse(text)
            print(f"DEBUG: The parsed_url that will be passed for the normal processing is '{parsed_url}'")
            print(f"DEBUG: The parsed_url.netloc that will be used for the normal processing is '{parsed_url.netloc}'")
            if parsed_url.scheme and parsed_url.netloc:
                if 'youtube' in parsed_url.netloc or 'youtu.be' in parsed_url.netloc:
                    content = process_youtube_link(parsed_url, RAPIDAPI_KEY)
                    text = f'{text} \n Please first fully understand the following transcript: """{content}""". Now make a short summary of the transcript and get ready for questions from the user.'
                elif parsed_url.netloc not in FORBIDDEN_DOMAINS:
                    content = process_generic_link(text)
                    text = f'{text} \n Please fully understand the following content from {parsed_url.netloc} and be ready for questions from the user: \n"""{content}"""'
                else:
                    text = f"The user shared this link: {text}. Please acknowledge it and say that you cannot work with it because it is not in the allowed domains."
            
            response = generate_response(text, current_user, S3_BUCKET, OPENAI_API_KEY, GOOGLE_API_KEY, TELEGRAM_BOT_TOKEN, chat_id, MODEL_NAME)
        else: 
            # text = f"Check in our current conversation and return: '{text}'"
            response = generate_response(text, current_user, S3_BUCKET, OPENAI_API_KEY, GOOGLE_API_KEY, TELEGRAM_BOT_TOKEN, chat_id, MODEL_NAME)

        print(f"DEBUG: The response from generate_response is '{response}")

        audio_warning = ""
        if BOT_AUDIO_RESPONSE and len(response) > MAX_RESPONSE_LENGTH_AUDIO:
            audio_warning = "<The text is too long so it will not be sent as audio...>\n\n"

        full_response = response + audio_warning

        if current_user == AUX_USERNAME:
            print(f"DEBUG: The current_user is AUX_USERNAME")
            for user_id in USERS_ALLOWED:
                user_chat_id = get_user_chat_id(user_id)
                if user_chat_id:
                    print(f"DEBUG: sending the message '{full_response}' to the user_chat_id '{user_chat_id}'")
                    send_message_to_bot(TELEGRAM_BOT_TOKEN, user_chat_id, full_response)
                    if BOT_AUDIO_RESPONSE and len(response) <= MAX_RESPONSE_LENGTH_AUDIO:
                        send_audio_to_bot(TELEGRAM_BOT_TOKEN, user_chat_id, response)
        else:
            send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, full_response)
            if BOT_AUDIO_RESPONSE and len(response) <= MAX_RESPONSE_LENGTH_AUDIO:
                send_audio_to_bot(TELEGRAM_BOT_TOKEN, chat_id, response)

        message_data = {
            "uuid": uuid,
            "message_id": message["message_id"],
            "from": message["from"],
            "chat": message["chat"],
            "date": timestamp, 
            "human_readable_date": datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            "text": text,
            "response": response
        }
        save_message_to_json(current_user, message_data, S3_BUCKET)
        return {
            'statusCode': 200,
            'body': json.dumps('Lambda finished OK.')
        }
    except KeyError as e:
        error_message = f"Error: Missing required field in the message: {str(e)}"
        print(f"DEBUG: {error_message}")
        if 'chat_id' in locals():
            send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, error_message)
        return {
            'statusCode': 400,
            'body': json.dumps(error_message)
        }
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        print(f"DEBUG: {error_message}")
        if 'chat_id' in locals():
            send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, error_message)
        return {
            'statusCode': 500,
            'body': json.dumps(error_message)
        }


def main():
    # Load environment variables from .env file if it exists
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from {env_path}")
    else:
        logger.info(".env file not found, using system environment variables")
    
    # Initialize model configuration for local development
    try:
        model_provider = os.environ.get('MODEL_PROVIDER', DEFAULT_MODEL_PROVIDER)
        gemini_model = os.environ.get('GEMINI_MODEL', DEFAULT_GEMINI_MODEL)
        openai_model = os.environ.get('OPENAI_MODEL', DEFAULT_OPENAI_MODEL)
        initialize_model_config(model_provider, gemini_model, openai_model)
        logger.info(f"Local model configuration loaded: {model_provider}")
    except Exception as e:
        logger.warning(f"Failed to load local model config: {e}. Using defaults.")
        initialize_model_config(DEFAULT_MODEL_PROVIDER, DEFAULT_GEMINI_MODEL, DEFAULT_OPENAI_MODEL)
    
    # Initialize AWS clients for local environment
    global ssm, s3_client, polly
    
    RESPOND_LAST_N_MESSAGES = 1
    context =''
    ssm = boto3.client('ssm', region_name=region)  # Adjust the region as needed.
    s3_client = boto3.client('s3', region_name=region)
    polly = boto3.client('polly')
    TELEGRAM_BOT_TOKEN = get_parameter(ssm, TELEGRAM_BOT_TOKEN_SSM_PATH)
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    response = requests.get(url)
    updates = response.json()
    if updates["result"] == []:
        print("No updates")
        return

    events = [] #NOTE: here I am creating a list of events but the lambda will only get one event with the info of one message. Each message triggers a webhook that triggers a lambda
    for result in updates["result"]:
        event = {
            "resource": "/{proxy+}",
            "path": "/your_bot_path",
            "httpMethod": "POST",
            "headers": {
                "Content-Type": "application/json"
            },
            "body": json.dumps(result),
            "isBase64Encoded": 'false'
        }
        events.append(event)

    for event in events[-RESPOND_LAST_N_MESSAGES:]:
        lambda_handler(event, context)


if __name__ == "__main__":
    # Only run main() when executed directly (not imported by Lambda)
    if not is_running_in_lambda():
        main()


