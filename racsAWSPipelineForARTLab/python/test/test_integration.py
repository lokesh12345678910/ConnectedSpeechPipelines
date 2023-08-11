import unittest
import os

class TestIntegration(unittest.TestCase):
    """This test checks that the output of the test from the pre-processing stage is the same as the input
    for the test of the processing stage. We want to make sure R is expecting the same thing we are outputting
    """

    def test_R_can_handle_current_python(self):
        """
        Ensure that the tests we are running for R use the exact same input
        as the output for Python
        """
        pre_processing_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Get the Python output directory
        python_end2end_dir = os.path.join(pre_processing_dir, 'data', 'end2end', 'short-file')
        python_output_dir = os.path.join(python_end2end_dir, 'output')
        
        # Get the R input directory
        app_dir = os.path.dirname(pre_processing_dir)
        r_data_dir = os.path.join(app_dir, "r-processing", "R", "tests", "testdata", "end2end", "short-file")
        r_input_dir = os.path.join(r_data_dir, "input")
        
        # Assert the files are the same
        python_files = os.listdir(python_output_dir)
        r_files = list(filter(lambda x: x != '.Rhistory', os.listdir(r_input_dir)))
        # Assert they have the same files
        self.assertListEqual(r_files, python_files)
        
        # For every file, assert they have the same data
        for file_name in python_files:
            with open(os.path.join(python_output_dir, file_name), 'r') as python_file, open(os.path.join(r_input_dir, file_name), 'r') as r_file:
                self.assertListEqual(python_file.read().splitlines(), 
                                     r_file.read().splitlines(), 
                                     f"The R file is not testing on the most recent version of the Python output files on the file. Change either the " + 
                                     "Python code or change the R data file and ensure R code can handle it with its testing suite. FILE: {file_name}")
 

if __name__ == '__main__':
    unittest.main()
