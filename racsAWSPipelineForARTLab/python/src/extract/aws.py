import logging

def extractAWSID(file_name: str) -> str:
    """Get the AWS ID for a given file

    Args:
        file_name (str): The file name we want to convert

    Returns:
        str: The AWS ID for the file
    """
    return file_name.split('_')[0]


def getLogger() -> logging.Logger:
    """
    AWS Lambda only lets you see your logger if you set it to INFO each time. This
    allows us to do it in one place.
    """
    # If it is AWS
    if logging.getLogger(__name__).hasHandlers():
        # If it is AWS Lambda, configure the AWS logger
        logging.getLogger(__name__).setLevel(logging.INFO)
        
    return logging.getLogger(__name__)