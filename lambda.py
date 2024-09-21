# Lambda function: serializeImageData
import json
import boto3
import base64

s3 = boto3.client('s3')

def serialize_image_data(event, context):
    """A function to serialize target data from S3"""

    # Get the S3 address from the Step Function event input
    key = event["s3_key"]
    bucket = event["s3_bucket"]

    # Download the data from S3 to /tmp/image.png
    s3.download_file(bucket, key, "/tmp/image.png")

    # Read the data from the file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read()).decode('utf-8')

    # Pass the data back to the Step Function
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


# Lambda function: classifyImageData
import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

# Replace with your actual deployed endpoint name
ENDPOINT = "image-classification-2024-09-20-17-06-22-080"

def classify_image_data(event, context):
    """A function to classify image data using SageMaker"""

    # Decode the image data
    image = base64.b64decode(event["body"]["image_data"])

    # Instantiate a Predictor
    predictor = sagemaker.Predictor(ENDPOINT)

    # Set the serializer to image/png
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction
    inferences = predictor.predict(image)

    # We return the data back to the Step Function    
    event["body"]["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event["body"])
    }


# Lambda function: filterInferences
import json

# Set the confidence threshold
THRESHOLD = 0.93

def filter_inferences(event, context):
    """A function to filter low confidence inferences"""

    # Grab the inferences from the event
    inferences = json.loads(event["body"]["inferences"])

    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any(float(inference) > THRESHOLD for inference in inferences)

    # If our threshold is met, pass our data back out of the Step Function, else raise an error
    if meets_threshold:
        return {
            'statusCode': 200,
            'body': json.dumps(event["body"])
        }
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")
