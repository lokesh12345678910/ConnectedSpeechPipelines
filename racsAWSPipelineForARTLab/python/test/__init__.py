# This is required to deal with our package structure. It allows us to run tests
import os
import sys
PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(PROJECT_PATH, "src")
sys.path.append(SOURCE_PATH)