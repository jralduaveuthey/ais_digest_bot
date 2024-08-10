import io
import time
import boto3
import json
import logging
from langchain import PromptTemplate, ConversationChain
import requests
from datetime import datetime
import hashlib
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMemory
from pydantic import BaseModel
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
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

#-------------------BOT SETTINGS-------------------#

DEFAULT_MODEL = "gpt-4o-mini"  # Use this as a fallback
MAX_TOKENS = 128000  # For GPT-4o-mini, the maximum token limit is 128,000 tokens
DEFAULT_ENCODING = "cl100k_base"  # This is used by gpt-3.5-turbo and gpt-4

prompt_template = """The following is a friendly conversation between a human and an AI. 
    The AI is talkative and provides lots of specific details from its context. 
    If the AI does not know the answer to a question, it truthfully says it does not know.
    Here is the history of the conversation that must be checked before answering any question: 
    {conversation_history}
    Now, the conversation continues:
    Human: {input}
    AI:"""

TELEGRAM_BOT_TOKEN_SSM_PATH ='/chatbot-ais-digest/TELEGRAM_BOT_TOKEN_AIS_Digest'
OPENAI_API_KEY_SSM_PATH = '/chatbot-ais-digest/OPENAI_API_KEY'
S3_BUCKET_SSM_PATH = '/chatbot-ais-digest/S3_BUCKET'
USERS_ALLOWED_SSM_PATH = '/chatbot-ais-digest/USERS_ALLOWED'
RAPIDAPI_KEY_SSM_PATH = '/chatbot-ais-digest/RAPIDAPI_KEY'
AUX_USERNAME_SSM_PATH = '/aux-chatbot-ais-digest/AUX_USERNAME'

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
ssm = boto3.client('ssm', region_name=region)  # Adjust the region as needed.
s3_client = boto3.client('s3', region_name=region)
polly = boto3.client('polly')



class ConversationHistoryMemory(BaseMemory, BaseModel):
    """Memory class for storing conversation history."""
    # Define list to store conversation history.
    conversation_history: Optional[List[Tuple[str, str]]] = []
    # Define key to pass conversation history into prompt.
    memory_key: str = "conversation_history"

    def clear(self):
        self.conversation_history = []

    @property
    def memory_variables(self) -> List[str]:
        """Define the variables we are providing to the prompt."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Load the memory variables, in this case the conversation history."""
        # Get the previous conversations.
        previous_conversations = self.conversation_history
        # Return combined conversations to put into context.
        return {self.memory_key: "\n".join(f"Human: {human}\nAI: {ai}" for human, ai in previous_conversations)}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        # Get the input and output texts.
        input_text = inputs[list(inputs.keys())[0]]
        output_text = outputs[list(outputs.keys())[0]]
        # Append the conversation to the history.
        self.conversation_history.append((input_text, output_text))


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
    """Retrieve a parameter by name from AWS Systems Manager Parameter Store."""
    print(f"Getting parameter '{name}' from the AWS Parameter Store.")
    response = ssm.get_parameter(Name=name, WithDecryption=True)
    return response['Parameter']['Value']

def send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, text):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    data = {"chat_id": chat_id, "text": text}
    requests.post(url, data=data)

def send_audio_to_bot(TELEGRAM_BOT_TOKEN, chat_id, text):
    try:
        response = polly.synthesize_speech(
            OutputFormat='mp3',
            Text=text,
            VoiceId=voice_name
        )
        if "AudioStream" in response:
            with closing(response["AudioStream"]) as stream:
                audio = stream.read()

        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendAudio"
        data = {"chat_id": chat_id}
        files = {"audio": io.BytesIO(audio)}
        requests.post(url, data=data, files=files)
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

def num_tokens_from_messages(messages: List[Union[BaseMessage, Tuple[str, str]]], model_name: str) -> int:
    encoding = get_encoding(model_name)
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        if isinstance(message, BaseMessage):
            content = message.content
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                role = "user"  # default to user for unknown message types
        elif isinstance(message, tuple):
            user_message, ai_response = message
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")
        
        num_tokens += len(encoding.encode(user_message))
        num_tokens += len(encoding.encode(ai_response))
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens

def trim_messages(
    messages: List[BaseMessage],
    max_tokens: int,
    token_counter: Callable[[List[BaseMessage]], int],
    strategy: str = "last",
    include_system: bool = True,
) -> List[BaseMessage]:
    if strategy not in ["last", "first"]:
        raise ValueError(f"Invalid strategy: {strategy}")
    
    system_message = next((m for m in messages if isinstance(m, SystemMessage)), None)
    non_system_messages = [m for m in messages if not isinstance(m, SystemMessage)]
    
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

def process_chunks(chunks: List[str], conversation: ConversationChain, TELEGRAM_BOT_TOKEN: str, chat_id: int) -> str:
    """Processes each chunk and returns the final response."""
    responses = []
    for i, chunk in enumerate(chunks):
        response = conversation.predict(input=f"Chunk {i+1}/{len(chunks)}: {chunk}")
        responses.append(response)
        
        # Send a progress update to the user
        progress_message = f"Processing chunk {i+1} of {len(chunks)}..."
        send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, progress_message)

    final_response = "\n\n".join(responses)
    return final_response

def handle_context_length(text: str, conversation, TELEGRAM_BOT_TOKEN: str, chat_id: int, model_name: str) -> str:
    try:
        return conversation.predict(input=text)
    except Exception as e:
        if "maximum context length" in str(e).lower():
            warning_message = ("The context was too long. Trimming conversation history to fit within the token limit.")
            send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, warning_message)
            
            all_messages = conversation.memory.conversation_history + [HumanMessage(content=text)]
            
            trimmed_messages = trim_messages(
                all_messages,
                max_tokens=MAX_TOKENS - 1000,  # Leave some room for the response
                token_counter=lambda msgs: num_tokens_from_messages(msgs, model_name),
                strategy="last",
                include_system=True
            )
            
            conversation.memory.conversation_history = trimmed_messages[:-1]  # Exclude the last message (current input)
            
            return conversation.predict(input=text)
        elif "You exceeded your current quota, please check your plan and billing details." in str(e):
            return "Sorry, I'm currently out of credits. Please try again later."
        else:
            raise e

def generate_response(text: str, username: str, S3_BUCKET: str, OPENAI_API_KEY: str, TELEGRAM_BOT_TOKEN: str, chat_id: int, model_name: str) -> str:
    conversation_history = load_conversation_history(username, S3_BUCKET)
    prompt = PromptTemplate(input_variables=["conversation_history", "input"], template=prompt_template)
    
    conversation = ConversationChain(
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=model_name, max_retries=1), 
        prompt=prompt, 
        verbose=True, 
        memory=ConversationHistoryMemory(conversation_history=conversation_history)
    )

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

def lambda_handler(event, context):
    try:
        MODEL_NAME = 'gpt-4o-mini'

        # Load tokens from Parameter Store
        TELEGRAM_BOT_TOKEN = get_parameter(ssm, TELEGRAM_BOT_TOKEN_SSM_PATH)
        OPENAI_API_KEY = get_parameter(ssm, OPENAI_API_KEY_SSM_PATH)
        S3_BUCKET = get_parameter(ssm, S3_BUCKET_SSM_PATH)
        USERS_ALLOWED = get_parameter(ssm, USERS_ALLOWED_SSM_PATH).split(',')
        AUX_USERNAME = get_parameter(ssm, AUX_USERNAME_SSM_PATH)
        RAPIDAPI_KEY = get_parameter(ssm, RAPIDAPI_KEY_SSM_PATH)

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

        if text.strip().lower().startswith("/new"):
            text = "(/new) Let's start a new conversation."
            response = generate_response(text, current_user, S3_BUCKET, OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, chat_id, MODEL_NAME)
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
            - Create a test question that puts what you’ve learned into a real-world context
            - Take a difficult question you found in a practice test and modify it so that it involves different variables or adds an extra step
            - Form a study group (user + you the AI assistant) and quiz each other (for some subjects, you can even debate the topic, with one side trying to prove that the other person is missing a point or understanding it incorrectly)

            Do not make any information up and respond only with information from the text. If it does not appear or you do not know then say so without making up information.
            """
            response = generate_response(text, current_user, S3_BUCKET, OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, chat_id, MODEL_NAME)
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

            Remember, your role is to provide supportive, non-judgmental prompts that encourage the user to think deeply and compassionately about themselves. Your prompts should be varied and cater to different aspects of the user’s life and inner experiences, helping them to continuously grow and improve without needing to respond directly to you.
            """
            response = generate_response(text, current_user, S3_BUCKET, OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, chat_id, MODEL_NAME)
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
            
            response = generate_response(text, current_user, S3_BUCKET, OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, chat_id, MODEL_NAME)
        else: 
            text = f"Check in our previous conversation and return: '{text}'"
            response = generate_response(text, current_user, S3_BUCKET, OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, chat_id, MODEL_NAME)

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


