import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
import boto3

# Initialize AWS clients
s3_client = boto3.client('s3')
sqs_client = boto3.client('sqs', region_name=os.environ['REGION_NAME'])


class Bot:

    def __init__(self, token, telegram_chat_url):
        # Create a new instance of the TeleBot class.
        self.telegram_bot_client = telebot.TeleBot(token)

        # Remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)

        # Set the webhook URL
        self.telegram_bot_client.set_webhook(
            url=f'{telegram_chat_url}/{token}/',
            timeout=60,
            certificate=open('/app/YOURPUBLIC.pem', 'r')
        )

        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        """
        Downloads the photos that are sent to the Bot to the `photos` directory (should exist)
        """
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type "photo" expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    def handle_message(self, msg):
        """Bot main message handler"""
        logger.info(f'Incoming message: {msg}')
        self.send_text(msg['chat']['id'], f'Your original message: {msg["text"]}')


class ObjectDetectionBot(Bot):
    def __init__(self, token, telegram_chat_url, s3_bucket_name, sqs_queue_url):
        super().__init__(token, telegram_chat_url)
        self.s3_bucket_name = s3_bucket_name
        self.sqs_queue_url = sqs_queue_url

    def upload_photo_to_s3(self, photo_path):
        # Upload the photo to S3
        s3_key = os.path.basename(photo_path)
        s3_client.upload_file(photo_path, self.s3_bucket_name, s3_key)
        logger.info(f'Uploaded {photo_path} to S3 bucket {self.s3_bucket_name} with key {s3_key}')
        return s3_key

    def send_job_to_sqs(self, s3_key, msg):
        # Send a job to the SQS queue
        message_body = {
            's3_key': s3_key,
            'chat_id': msg['chat']['id'],
        }

        response = sqs_client.send_message(
            QueueUrl=self.sqs_queue_url,
            MessageBody=str(message_body)
        )

        logger.info(f'Sent job to SQS queue {self.sqs_queue_url} with response {response}')

    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')
        if self.is_current_msg_photo(msg):
            photo_path = self.download_user_photo(msg)
            s3_key = self.upload_photo_to_s3(photo_path)
            self.send_job_to_sqs(s3_key, msg)
            self.send_text(msg['chat']['id'], 'Your image is being processed. Please wait ...')
        else:
            super().handle_message(msg)
