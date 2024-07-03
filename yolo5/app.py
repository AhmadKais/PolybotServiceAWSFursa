import time
from decimal import Decimal
from pathlib import Path
from detect import run
import yaml
from loguru import logger
import os
import boto3
import requests

images_bucket = os.environ['BUCKET_NAME']
queue_name = os.environ['SQS_QUEUE_NAME']
region_name = os.environ['REGION_NAME']
dynamodb_table_name = os.environ['DYNAMODB_TABLE_NAME']
polybot_endpoint = os.environ['POLYBOT_ENDPOINT']

sqs_client = boto3.client('sqs', region_name=region_name)
s3_client = boto3.client('s3', region_name=region_name)
dynamodb = boto3.resource('dynamodb', region_name=region_name)
table = dynamodb.Table(dynamodb_table_name)

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']


def download_image_from_s3(bucket, img_name, local_path):
    s3_client.download_file(bucket, img_name, local_path)


def upload_image_to_s3(bucket, local_path, s3_path):
    s3_client.upload_file(local_path, bucket, s3_path)


def convert_to_decimal(data):
    if isinstance(data, list):
        return [convert_to_decimal(i) for i in data]
    elif isinstance(data, dict):
        return {k: convert_to_decimal(v) for k, v in data.items()}
    elif isinstance(data, float):
        return Decimal(str(data))
    else:
        return data


def consume():
    while True:
        response = sqs_client.receive_message(QueueUrl=queue_name, MaxNumberOfMessages=1, WaitTimeSeconds=5)

        if 'Messages' in response:
            message = response['Messages'][0]['Body']
            receipt_handle = response['Messages'][0]['ReceiptHandle']
            prediction_id = response['Messages'][0]['MessageId']

            logger.info(f'prediction: {prediction_id}. start processing')

            msg_data = yaml.safe_load(message)
            img_name = msg_data['s3_key']
            chat_id = msg_data['chat_id']
            original_img_path = f"/tmp/{img_name}"

            download_image_from_s3(images_bucket, img_name, original_img_path)
            logger.info(f'prediction: {prediction_id}. download from s3 finished')

            detect_img_path = f"/tmp/detect_{img_name}"
            try:
                run(
                    weights="yolov5s.pt",
                    source=original_img_path,
                    data="data/coco128.yaml",
                    project="/tmp",
                    name=f"detect_{img_name}",
                    exist_ok=True
                )

                logger.info(f'prediction: {prediction_id}. detect finished')
            except Exception as e:
                logger.error(f'prediction: {prediction_id}. detect failed. err: {e}')

            labels_path = list(Path("/tmp").rglob(f'detect_{img_name}/*.txt'))[0]

            with open(labels_path) as f:
                data = f.readlines()

            logger.info(f'prediction: {prediction_id}. labels read')
            results = []

            for row in data:
                class_id, conf, *_ = row.split()
                class_id = int(class_id)
                conf = float(conf)
                results.append({'class': names[class_id], 'confidence': conf})

            logger.info(f'prediction: {prediction_id}. labels processed')
            table.put_item(Item=convert_to_decimal({
                'prediction_id': prediction_id,
                'chat_id': chat_id,
                'labels': results
            }))

            logger.info(f'prediction: {prediction_id}. saved to dynamodb')

            # Send POST request to Polybot service with prediction_id
            try:
                response = requests.post(f"{polybot_endpoint}/results", params={'predictionId': prediction_id})
                response.raise_for_status()
                logger.info(f'prediction: {prediction_id}. result sent to Polybot service')
            except requests.exceptions.RequestException as e:
                logger.error(f'prediction: {prediction_id}. failed to send result to Polybot service: {e}')

            sqs_client.delete_message(QueueUrl=queue_name, ReceiptHandle=receipt_handle)
            logger.info(f'prediction: {prediction_id}. delete message from queue')
        else:
            logger.info('no messages to consume')
            time.sleep(5)


if __name__ == '__main__':
    logger.info('start consuming')
    consume()
