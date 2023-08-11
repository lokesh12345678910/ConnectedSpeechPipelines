## Setup
1. Install Miniconda 
2. Set up your anaconda environment. Run the following from the `pre-processing` file
   ```sh
   conda create --name racs python=3.8
   conda activate racs
   conda install pip
   cd ConnectedSpeechPipelines/racsAWSPipelineForARTLab/
   pip install -r python/src/requirements.txt
   pip install -r python/test/requirements.txt
   conda install -c conda-forge ffmpeg
   pip install spacy_transformers
   pip install polyglot
   conda install -c conda-forge pyicu
   pip install pycld2
   pip install morfessor
   conda install -c anaconda scipy
   ```
   It might feel a little weird that we are using `pip` in `conda`, but multiple programmers have concluded this is the only way; please lmk if you work out how to avoid using pip 
5. Anytime you want to run the code, make sure you source your environment by running `conda activate racs`.
6. You need to install these Python extras that are needed for some of the analysis. Run
    ```sh
    python -m spacy download en_core_web_trf
    python -m nltk.downloader brown stopwords names cmudict
    ```

## Running Locally
There are a few ways to run this locally.

### Run from the terminal
First, you need to make sure you are in the right environment. Run
```sh
conda activate racs
```
to get into the correct Python environment.

In the `python/src/` folder, run 
```sh
python app.py --input-dir <INPUT FOLDER WITH DATA YOU WANT TO ANALYZE> \
    --output-dir <OUTPUT FOLDER> \
    --static-dir <PATH TO STATIC FILE> \
    --output-file-prefix <NAME OF FILE PREFIX FOR OUTPUT FILES> \
    --output-file-type <csv>
```
Example command: python /work/09424/smgrasso1/ls6/racsAWSPipeline/python/src/app.py --input-dir ./racsTrialFiles --static-dir /work/09424/smgrasso1/ls6/racsAWSPipeline/data/end2end/static/LingFeatData/ --output-file-prefix trial --output-file-type csv

Example command on Lisa's tacc work directory: python ConnectedSpeechPipelines/racsAWSPipelineForARTLab/python/src/app.py --input-dir ./racsTrialFiles  --output-dir ./ --static-dir ConnectedSpeechPipelines/racsAWSPipelineForARTLab/data/end2end/static/LingFeatData/ --output-file-prefix racsPipelineTrial --output-file-type csv 


When running, the pipeline will create 3 temporary directories in your running directory (wavFiles, monoWavFiles, trimmedMonoWavFiles); these will be deleted once the pipeline finishes

More info on these commands can be found in the file [app.py](python/src/app.py).

### Run from other Python code
You can pull the function `run_pipeline` directly into your Python code. More information on the `run_pipeline(...)` command can be found in the file [pipeline.py](python/src/extract/pipeline.py).

