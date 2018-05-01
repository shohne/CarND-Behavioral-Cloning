import boto3
import botocore
import zipfile
import sys
import threading

BUCKET_NAME = 'carnd-dataset-hohne'
KEY = 'dataset_carnd_behavioral_cloning.zip'

class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._seen_so_far = 0
        self._lock = threading.Lock()
    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            sys.stdout.write(
                "\r%s --> %s bytes transferred" % (
                    self._filename, self._seen_so_far))
            sys.stdout.flush()

def download_dataset_file(dst_directory):
    s3 = boto3.client('s3')
    print ('downloading dataset file...')
    s3.download_file(BUCKET_NAME, KEY, 'data.zip', Callback=ProgressPercentage("data.zip"))
    zip_ref = zipfile.ZipFile('data.zip', 'r')
    zip_ref.extractall()
    zip_ref.close()
