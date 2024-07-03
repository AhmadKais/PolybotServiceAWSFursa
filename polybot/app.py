import flask
from flask import request
import os
import boto3
from bot import ObjectDetectionBot
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

app = flask.Flask(__name__)


# Load TELEGRAM_TOKEN value from AWS Secrets Manager
def get_secret(secret_name):
    try:
        region_name = os.getenv('AWS_REGION', 'us-east-1')  # Default to us-east-1 if not set
        client = boto3.client('secretsmanager', region_name=region_name)
        response = client.get_secret_value(SecretId=secret_name)
        secret = response['SecretString']
        return secret
    except (NoCredentialsError, PartialCredentialsError) as e:
        raise Exception(f"Credentials error when accessing Secrets Manager: {e}")
    except Exception as e:
        raise Exception(f"An error occurred when retrieving the secret: {e}")


TELEGRAM_TOKEN = get_secret(os.environ['TELEGRAM_TOKEN_SECRET_NAME'])  # Get secret name from environment

TELEGRAM_APP_URL = os.environ['TELEGRAM_APP_URL']
S3_BUCKET_NAME = os.environ['S3_BUCKET_NAME']
SQS_QUEUE_NAME = os.environ['SQS_QUEUE_NAME']
DYNAMODB_TABLE_NAME = os.environ['DYNAMODB_TABLE_NAME']
AWS_REGION = os.environ['AWS_REGION']


@app.route('/', methods=['GET'])
def index():
    return 'Ok'


@app.route(f'/{TELEGRAM_TOKEN}/', methods=['POST'])
def webhook():
    req = request.get_json()
    bot.handle_message(req['message'])
    return 'Ok'


@app.route(f'/results', methods=['POST'])
def results():
    prediction_id = request.args.get('predictionId')

    # Retrieve results from DynamoDB using prediction_id and send to the end-user
    dynamodb = boto3.resource('dynamodb', region_name=AWS_REGION)
    table = dynamodb.Table(DYNAMODB_TABLE_NAME)  # Get table name from environment

    try:
        response = table.get_item(
            Key={'prediction_id': prediction_id}
        )
        item = response['Item']
        chat_id = item['chat_id']
        labels = item['labels']
        text_results = format_results(labels)
        bot.send_text(chat_id, text_results)
        return 'Ok'
    except Exception as e:
        return f'Error retrieving results: {e}', 500


@app.route(f'/loadTest/', methods=['POST'])
def load_test():
    req = request.get_json()
    bot.handle_message(req['message'])
    return 'Ok'


def format_results(labels):
    formatted_results = ""
    for label in labels:
        formatted_results += f"Class: {label['class']}, Confidence: {label['confidence']}\n"
    return formatted_results


if __name__ == "__main__":
    bot = ObjectDetectionBot(
        TELEGRAM_TOKEN,
        TELEGRAM_APP_URL,
        S3_BUCKET_NAME,
        SQS_QUEUE_NAME
    )
    app.run(host='0.0.0.0', port=8443, ssl_context=('/app/YOURPUBLIC.pem', '/app/YOURPRIVATE.key'))
