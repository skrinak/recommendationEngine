import os
import io
import boto3
import json
import csv

# grab environment variables for model endpoint name
ENDPOINT_NAME_RATING = os.environ['ENDPOINT_NAME_RATING']
ENDPOINT_NAME_REC = os.environ['ENDPOINT_NAME_REC']
# declare and initiate global variable for SageMaker runtime
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):

    data = json.loads(json.dumps(event))
    input = data['data'] # data to pass onto model endpoint
    pred_type = data ['type'] # if rating, it calls rating model. Otherwise, it calls recommendation model. 
    
    print(type(input))
    print(pred_type)
    
    if pred_type == 'rating':
        ENDPOINT_NAME = ENDPOINT_NAME_RATING
    else:
        ENDPOINT_NAME = ENDPOINT_NAME_REC
        
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='application/json',
                                       Body=input)

    result = json.loads(response['Body'].read().decode())
    print(result)
    pred = result['predictions'][0]['scores']
    return_str = ''
    
    if pred_type == 'rating':
        return_str = "The User will likely rate this movie to "  + str(pred) 
    else: # recommendation
        if pred[1] > 0.5:
            return_str = "We recommend the user this movie."
        else: 
            return_str = "We DO NOT recommend the user this movie."
        
    return return_str
    
    
  