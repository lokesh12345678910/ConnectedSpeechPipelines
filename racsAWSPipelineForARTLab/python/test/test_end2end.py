import unittest
import os
from backports import tempfile

from extract.pipeline import run_pipeline

class TestEnd2End(unittest.TestCase):
        
    def test_short_end2end(self):
        """
        Test the short audio file that we have
        If you want to add more tests, copy this name (ensure test_ is before it), 
        add a new directory at the same level as "short-file", and exchange that for the below line
        """
        self.run_end2end('short-file')
                        
                        
    def run_end2end(self, folder_with_input_output: str):
        """This is an end to end test that ensures the pipeline runs. 
        
        Once the pipeline has run, it checks to see if there are any differences in the file output. If there are, 
        it will require that you acknowledge the changes. This ensures that you don't make unexpected changes in the
        pipeline.
        """
        # Get the path of this specific file
        this_dir = os.path.dirname(os.path.abspath(__file__))
        pre_processing_dir = os.path.dirname(os.path.dirname(this_dir))
        end2end_dir = os.path.join(pre_processing_dir, 'data', 'end2end')
        
        # Get the paths to our input and outputs
        input_dir = os.path.join(end2end_dir, folder_with_input_output, 'input')
        static_dir = os.path.join(end2end_dir, 'static', 'LingFeatData')
        expected_dir = os.path.join(end2end_dir, folder_with_input_output, 'output')
        
        # Create a temporary directory for us to write the pipeline files to
        with tempfile.TemporaryDirectory() as temp_dir:
            
        
            # If we don't have any files in the expected output directory (or the directory doesn't exist), set the actual 
            # output directory to the expected one and write all files there
            if not os.path.exists(expected_dir) or len(os.listdir(expected_dir)) == 0:
                actual_dir = expected_dir
                
                # If the folder doesn't exist, make it
                if not os.path.exists(expected_dir):
                    os.makedirs(expected_dir)
            # Otherwise, if we have expected output, we want the pipeline to write files to the actual directory
            else:
                actual_dir = temp_dir
            
            # Run the pipeline
            run_pipeline(
                input_directory=input_dir,
                static_files_directory=static_dir,
                output_directory=actual_dir,
                output_file_prefix="test",
                output_file_type='json',
            )
            
            # If we are testing to old output, confirm we are outputting identical files
            if actual_dir != expected_dir:
                expected_dir_files = os.listdir(expected_dir)
                actual_dir_files = os.listdir(actual_dir)
                expected_dir_files.sort()
                actual_dir_files.sort()
                # Make sure all the files in the expected directory are in the actual directory and no more
                self.assertListEqual(expected_dir_files, actual_dir_files, "The actual directory and expected directory should output identical files")
                
                # For each file in the expected directory, assert the contents are the same
                for file in expected_dir_files:
                    with open(os.path.join(expected_dir, file), 'r') as expected_file_contents, open(os.path.join(actual_dir, file), 'r') as actual_file_contents:
                        # If this is failing, you can delete all the files in data/end2end/output. Make sure to check why the files are different in git before
                        # deploying to AWS Lambda. Otherwise, you may be making a change that shouldn't happen
                        self.assertListEqual(expected_file_contents.read().splitlines(), actual_file_contents.read().splitlines(), f"The contents of the file in the expected and actual file are not the same for {file}")
            
                    
if __name__ == '__main__':
    unittest.main()