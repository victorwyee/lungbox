# -*- coding: utf-8 -*-
"""Functions for interacting with AWS"""

from io import BytesIO

import os
import boto3
import botocore
import pydicom
import pandas as pd

S3_CLIENT = boto3.client('s3')      # low-level functional API
S3_RESOURCE = boto3.resource('s3')  # higher-level OOO API

def list_bucket_contents(bucket_name):
    """List all contents from all buckets (using default configuration)."""
    obj_paths = []
    s3 = boto3.resource('s3')
    for obj in s3.Bucket(name=bucket_name).objects.all():
        obj_paths.append(os.path.join(obj.bucket_name, obj.key))
    return obj_paths

def parse_dicom_image_list(bucket_name, verbose=False):
    """Get list of DICOMs from S3 bucket and parse train/test and patientId."""
    # TODO: Not efficient. Find way not to do this in a loop.
    image_df = pd.DataFrame(columns=['path', 'subdir', 'patient_id'])
    for obj in S3_RESOURCE.Bucket(name=bucket_name).objects.all():
        path = os.path.join(obj.bucket_name, obj.key)
        if path.endswith('.dcm'):
            if verbose:
                print(path)
            path_parts = path.split('/')
            image_df = image_df.append({'path': path,
                                        # either 'train' or 'test'
                                        'subdir': path_parts[3].split('_')[2],
                                        'patient_id': path_parts[-1].split('.')[0]},
                                       ignore_index=True)
    return image_df

def read_s3_df(bucket_name, file_key):
    """Read CSV from S3 adn return a pandas dataframe."""
    try:
        obj = S3_CLIENT.get_object(Bucket=bucket_name, Key=file_key)
        return pd.read_csv(obj['Body'])
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise

def get_s3_dcm(bucket, dirpath, patient_id):
    """Read DICOM from S3"""
    obj = S3_CLIENT.get_object(Bucket=bucket,
                               Key=dirpath + '/%s.dcm' % patient_id)
    return pydicom.read_file(BytesIO(obj['Body'].read()))

def parse_training_labels(train_box_df, train_image_dirpath):
    """
    Method to read a CSV file (Pandas dataframe) and parse the
    data into the following nested dictionary:

      parsed = {
        'patientId-00': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia,
            'boxes': list of box(es)
        },
        'patientId-01': {
            'dicom': path/to/dicom/file,
            'label': either 0 or 1 for normal or pnuemonia,
            'boxes': list of box(es)
        }, ...
      }

    References:
    * Exploratory Data Analysis, by Peter Chang, MD.
        - Available at https://www.kaggle.com/peterchang77/exploratory-data-analysis/notebook
        - Retrieved 2018-09-19
    """
    # Define lambda to extract coords in list [y, x, height, width]
    def extract_box(row):
        return [row['y'], row['x'], row['height'], row['width']]

    parsed = {}
    for n, row in train_box_df.iterrows():
        # Initialize patient entry into parsed
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom_s3_key': train_image_dirpath + '%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed
