# -*- coding: utf-8 -*-
"""Data retrieval and pre-processing utilities."""

import os
import boto3
import botocore
from botocore.client import Config
from io import BytesIO
import pydicom
import pandas as pd
# import streamlit as st

config = Config(connect_timeout=50, read_timeout=70)
S3_CLIENT = boto3.client('s3', config=config)      # low-level functional API
S3_RESOURCE = boto3.resource('s3', config=config)  # higher-level OOO API


# @st.cache
def list_bucket_contents(bucket):
    """List all contents from all buckets (using default configuration)."""
    obj_paths = []
    s3 = boto3.resource('s3')
    for obj in s3.Bucket(name=bucket).objects.all():
        obj_paths.append(os.path.join(obj.bucket, obj.key))
    return obj_paths


# @st.cache
def parse_dicom_image_list(bucket, subdir='train', limit=None, verbose=False):
    """Get list of DICOMs from S3 bucket and parse train/test and patientId."""
    # TODO: SLOW. Not efficient. Find way not to do this in a loop.
    print("Retrieving image list...")
    image_df = pd.DataFrame(columns=['path', 'subdir', 'patient_id'])
    for obj in S3_RESOURCE.Bucket(name=bucket).objects.all():
        path = os.path.join(obj.bucket_name, obj.key)
        if path.endswith('.dcm'):
            if verbose:
                print(path)
            path_parts = path.split('/')
            subdir_part = path_parts[3].split('_')[2]
            if subdir_part == subdir:
                image_df = image_df.append({
                    'path': path,
                    'subdir': subdir_part,  # train or test
                    'patient_id': path_parts[-1].split('.')[0]},
                    ignore_index=True)
        if limit is not None and (len(image_df.index) == limit):
            break
    return image_df


# @st.cache
def read_s3_df(bucket, file_key):
    """Read CSV from S3 and return a pandas dataframe."""
    try:
        obj = S3_CLIENT.get_object(Bucket=bucket, Key=file_key)
        return pd.read_csv(obj['Body'])
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise


# @st.cache
def get_s3_dcm(bucket, file_key):
    """Read DICOM from S3"""
    obj = S3_CLIENT.get_object(Bucket=bucket, Key=file_key)
    return pydicom.read_file(BytesIO(obj['Body'].read()))


# @st.cache
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
                'dicom_s3_key': train_image_dirpath + '/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed
