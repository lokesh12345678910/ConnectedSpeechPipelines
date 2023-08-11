import argparse
import urllib.parse
import os
import uuid
import json
    
from extract.pipeline import run_pipeline
from extract.enums import FileType
from extract.aws import getLogger

    
logger = getLogger()

def lambda_handler(event, context):
    """
    This function is what is called from Lambda.
    It:
    - Downloads the s3 file that triggered the Lambda function
    - Runs the pre-processing pipeline on it
    - Uploads all files (with associated prefix) to the output section in S3
    
    If you are working on the ML code, you likely can skip this part and go into extract/pipeline.py file. That is where the code starts for ML stuff.
    """
    import boto3
    
    logger.info("Starting handler")
    
    # Get the object from the event and show its content type
    s3_event = json.loads(event['Records'][0]['Sns']['Message'])
    bucket = s3_event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(s3_event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    
    # FUNCTION_DIR is set in the Dockerfile
    # Static files are in Dockerfile. If you want to change them, go there and then re-deploy the image
    static_directory = os.path.join(os.environ['FUNCTION_DIR'], "data", "static")
    
    # Data we write to needs to be in /tmp because of lambda rules
    # Additionally, lambda has access to another lambda's /tmp files if it was recently run so we need to
    # generate a unique ID for this specific folder
    tmp_parent_path = os.path.join(os.environ["WRITABLE_DIR"], str(uuid.uuid1()))
    local_input_directory = os.path.join(tmp_parent_path, "input")
    local_output_directory = os.path.join(tmp_parent_path, "output")
    
    # Make all the folders above
    os.mkdir(tmp_parent_path)
    os.mkdir(local_input_directory)
    os.mkdir(local_output_directory)
    
    # Get file name
    file_name = os.path.basename(key)
    file_prefix = os.path.splitext(file_name)[0] # Strip off the extension
    
    # Download the input data we are using
    s3 = boto3.client('s3')
    logger.info(f"Downloading audio file: {file_prefix}")
    s3.download_file(bucket, key, os.path.join(local_input_directory, file_name))
    
    # Run the pipeline
    logger.info(f"Starting the ML pipeline: {file_prefix}")
    run_pipeline(input_directory = local_input_directory, 
                 static_files_directory = static_directory,
                 output_directory = local_output_directory,
                 output_file_prefix = file_prefix,
                 output_file_type = FileType.csv)
    
    # Upload the file to the S3 output directory
    s3_output_key = f"pre-processed/{file_prefix}"
    for output_file in os.listdir(local_output_directory):
        logger.info(f"Uploading file to S3: {output_file}: {file_prefix}")
        s3.upload_file(os.path.join(local_output_directory, output_file), bucket, os.path.join(s3_output_key, output_file))
        
    
    logger.info(f"Pipeline successful: {file_prefix}")
    
    sns_topic = "r_processing-trigger-notification"
    logger.info(f"Triggering next stage of pipeline by writing to SNS topic {sns_topic}")
    
    sns = boto3.client('sns')
    # Get the ARN
    sns_topic_arn = [x['TopicArn'] for x in sns.list_topics()['Topics'] if sns_topic in x['TopicArn']][0]
    # Send a message with the file prefix
    sns.publish(
        TargetArn=sns_topic_arn,
        Message=json.dumps({'default': json.dumps({
            "key": s3_output_key,
            "bucket": bucket
            })}),
        MessageStructure='json'
    )
    logger.info(f"Successfully published SNS topic {sns_topic}")
    return {
        "file": key
    }
    
    
# If we call the file directly, ensure tehre are arguments added
if __name__ == '__main__':
    # Create the args parser
    parser = argparse.ArgumentParser(
                    prog = 'racs-pre-processing',
                    description = 'Run the RACS pre-processing app')
    
    parser.add_argument('--input-dir', 
                        type=str, 
                        help='The input directory with the files that you want evaluated. This should contain the audio files',
                        required=True)
    
    parser.add_argument('--static-dir', 
                        type=str, 
                        help='The directory that holds the static files that are evaluated. Example: it holds the phoneme file',
                        required=True)
    
    parser.add_argument('--output-dir', 
                        type=str, 
                        help='The output directory where you want the feature results written',
                        required=True)

    parser.add_argument('--output-file-prefix', 
                        type=str, 
                        help='The prefix for the output files',
                        required=True)
    
    parser.add_argument('--output-file-type',
                        required=True,
                        default='csv',
                        choices=[x.name for x in FileType],
                        help="The output file format")
    
    
    args = parser.parse_args()
    
    # Run the pipeline
    run_pipeline(input_directory=args.input_dir,
                 static_files_directory=args.static_dir,
                 output_directory=args.output_dir,
                 output_file_prefix=args.output_file_prefix,
                 output_file_type=FileType[args.output_file_type])