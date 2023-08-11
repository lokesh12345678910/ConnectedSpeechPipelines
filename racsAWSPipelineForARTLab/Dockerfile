FROM ubuntu@sha256:7a57c69fe1e9d5b97c5fe649849e79f2cfc3bf11d10bbd5218b4eb61716aebe6

# The location of all of our code + where we will download data
ENV FUNCTION_DIR="/home/code"
# This lets the code decide where to write since Lambda doesn't let you write anywhere except in the temp directory
ENV WRITABLE_DIR="/tmp"
# Another thing to help them decide where to write
ENV HOME="/tmp"
# This is for NLTK which downloads data and is needed for the model
ENV NLTK_DATA="/home/code"

# This was to help install Python
ENV DEBIAN_FRONTEND=noninteractive

# Create the directory that will hold all of our code
# + make a directory for the output data which will later be uploaded to S3
RUN mkdir ${FUNCTION_DIR} \
    && mkdir ${FUNCTION_DIR}/data
WORKDIR ${FUNCTION_DIR}

# Install a bunch of stuff including python and pip
RUN apt update \
    && apt install -y build-essential software-properties-common libx11-dev curl ffmpeg default-jre\
    && apt-get install -y ninja-build \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y --no-install-recommends python3.8 \
    && apt install -y python3.8-distutils python3.8-dev \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.8 get-pip.py

# Copy our requirements file over & install. Do this before the other code so we don't have to
# reinstall requirements every time we update code
# Additionally, download the something one of the packages needs
COPY python/src/requirements.txt .
RUN pip3.8 install -r requirements.txt \
    # This is required in our code but doesn't have a very clean way to install via pip
    && python3.8 -m spacy download en_core_web_lg \
    # This is required for splat
    && python3.8 -m nltk.downloader -d ${NLTK_DATA} brown stopwords names cmudict

# Install the AWS Lambda runtime interface and boto3
RUN pip3.8 install --target ${FUNCTION_DIR} awslambdaric boto3

# Expose the port so the external lambda stuff can reach the state of our function
EXPOSE 8080

# Copy over the rest of the code
COPY python/src ${FUNCTION_DIR}

# Copy over the static data
COPY data/end2end/static/LingFeatData ${FUNCTION_DIR}/data/static

ENTRYPOINT [ "python3.8", "-m", "awslambdaric" ]
CMD [ "app.lambda_handler" ] 