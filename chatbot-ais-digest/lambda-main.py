import io
import time
import boto3
import json
import logging
from langchain import PromptTemplate, ConversationChain
import requests
from datetime import datetime
import hashlib
from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMemory
from pydantic import BaseModel
from typing import List, Dict, Any, Tuple, Optional
from contextlib import closing
import os
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse

#-------------------BOT SETTINGS-------------------#

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

def process_generic_link(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching URL {url}: {str(e)}")
        return f"Couldn't retrieve content from {url}. Error: {str(e)}"

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

    if "New Comment Submit" in text: #so tjat the comments are not included
        text = text[:text.index("New Comment Submit")] 
    return text

def process_youtube_link(url):
    video_id = url.split("v=")[1]
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        return f"Couldn't retrieve transcript: {str(e)}"
    
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

def generate_response(text, username, S3_BUCKET, OPENAI_API_KEY):
    # Load conversation history from the JSON file
    conversation_history = load_conversation_history(username, S3_BUCKET)

    # # Update conversation_history to keep only the messages containing the last 3000 charachters...so it does not hit token limit and so that it is cheaper
    # total_length = sum(len(x[0]) + len(x[1]) for x in conversation_history)# calculate the total length
    # while total_length > 3000:# keep popping from the front of the list until total length <= 3000
    #     msg = conversation_history.pop(0)
    #     total_length -= len(msg[0]) + len(msg[1])

    print(f">>>>>>>The conversation_history is '{conversation_history}")
    prompt = PromptTemplate(input_variables=["conversation_history", "input"], template=prompt_template)
    
    # Use the memory class
    conversation = ConversationChain(
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-4o-mini'), 
        prompt=prompt, 
        verbose=True, 
        memory=ConversationHistoryMemory(conversation_history=conversation_history)
    )
    if "New Comment Submit" in text:
        text = text[:text.index("New Comment Submit")]
        text = text.strip()
    # response = conversation.predict(input="what is my name?")
    response = conversation.predict(input=text)
    return response

def load_conversation_history(username, S3_BUCKET):
    filename = f"userlogs/{username}.json"
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=filename)
        data = json.loads(response['Body'].read().decode('utf-8'))
        full_history = [(item['text'], item['response']) for item in data]
        
        # Find the index of the last "/new" message
        last_new_index = -1
        for i, (text, _) in enumerate(reversed(full_history)):
            if text.strip().lower().startswith("/new"):
                last_new_index = len(full_history) - i - 1
                break
        
        # If "/new" was found, return the history after it, otherwise return the full history
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


def lambda_handler(event, context):
    try:
        # Load tokens from Parameter Store
        TELEGRAM_BOT_TOKEN = get_parameter(ssm, TELEGRAM_BOT_TOKEN_SSM_PATH)
        OPENAI_API_KEY= get_parameter(ssm, OPENAI_API_KEY_SSM_PATH)
        S3_BUCKET= get_parameter(ssm, S3_BUCKET_SSM_PATH)
        USERS_ALLOWED= get_parameter(ssm, USERS_ALLOWED_SSM_PATH).split(',')

        print(f"The event received is: {event}")
        message = json.loads(event['body'])['message']
        chat_id = message['chat']['id']
        current_user = message['chat']['username']
        if current_user not in USERS_ALLOWED:
            return {
                'statusCode': 200,
                'body': json.dumps('Sorry, but you first need to register to use this chatbot.')
            }

        if 'voice' in message and message['voice']['mime_type'] == 'audio/ogg': #the user sends an audio
            api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
            # get file path from telegram
            file_id = message['voice']['file_id']
            response = requests.get(api_url + "/getFile", params={"file_id": file_id}).json()
            file_path = response["result"]["file_path"]
            
            # download the audio file
            audio_data = requests.get(f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}").content
            
            # save audio data to S3
            audio_file = io.BytesIO(audio_data)
            s3 = boto3.client('s3')
            s3.put_object(Bucket=S3_BUCKET, Key='temp_audio.ogg', Body=audio_file)

            # use AWS Transcribe service
            TranscriptionJob = 'MyTranscriptionJob4Chatbot'+str(int(time.time()))
            transcribe = boto3.client('transcribe')
            transcribe.start_transcription_job(
                TranscriptionJobName=TranscriptionJob,
                Media={'MediaFileUri': f's3://{S3_BUCKET}/temp_audio.ogg'},
                MediaFormat='ogg',
                LanguageCode='en-US',
            )

            # wait for the job to finish
            while True:
                status = transcribe.get_transcription_job(TranscriptionJobName=TranscriptionJob)
                if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                    break
                time.sleep(5)

            # get the transcript text from the result
            transcript_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            transcript_result = requests.get(transcript_url).json()
            text = transcript_result['results']['transcripts'][0]['transcript']    
            send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, f"My transcript of your audio is: {text}")

        else: #the user typed something
            text = message['text']

        timestamp = message['date']
        uuid = generate_uuid(timestamp, text)
        print(f"The text received is: {text}")
        # print(f"Received message from {chat_id}: {message}")
        if text[-1].isalpha():  # adding a dot, without a dot ChatGPT will try to complete the sentence.
            text = text + "."
        if text.strip().lower().startswith("/new"):
            text = text + ". Let's start a new conversation."

        # Check if the text is a link
        parsed_url = urlparse(text)
        if parsed_url.scheme and parsed_url.netloc:
            if 'youtube' in parsed_url.netloc or 'youtu.be' in parsed_url.netloc:
                content = process_youtube_link(text)
                text = f'Please first fully understand the following transcript: """{content}""" . Now make a short summary of the transcript and get ready for questions from the user.'
            elif parsed_url.netloc not in FORBIDDEN_DOMAINS:
                content = process_generic_link(text)
                text = f"Please fully understand the following content from {parsed_url.netloc} and be ready for questions from the user: {content}"
            else:
                text = f"The user shared this link: {text}. Please acknowledge it and say that you cannot work with it because it is not in the allowed domains."
        

        response = generate_response(text, current_user, S3_BUCKET, OPENAI_API_KEY)

        # Check if audio response is enabled and if the response is too long
        audio_warning = ""
        if BOT_AUDIO_RESPONSE and len(response) > MAX_RESPONSE_LENGTH_AUDIO:
            audio_warning = "<The text is too long so it will not be sent as audio...>\n\n"

        # Combine the audio warning (if any) with the response
        full_response = response + audio_warning

        # Send the text response
        send_message_to_bot(TELEGRAM_BOT_TOKEN, chat_id, full_response)

        # Send audio only if it's enabled and the response is not too long
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
    except Exception as e:
        logging.error(str(e))
        return {
            'statusCode': 400,
            'body': json.dumps(f'My error: {e}')
        }



