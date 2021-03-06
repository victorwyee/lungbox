# -*- coding: utf-8 -*-
"""Data retrieval and pre-processing utilities."""

import os
import sys
import boto3
import botocore
from botocore.client import Config
from io import BytesIO
import pydicom
import pandas as pd
from pathlib import Path

try:
    script_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_path = "/projects/lungbox/src"
    sys.path.append(script_path)
sys.path.append(script_path + "/Mask_RCNN")


def list_bucket_contents(bucket):
    """List all contents from all buckets (using default configuration)."""
    obj_paths = []
    s3 = boto3.resource('s3')
    for obj in s3.Bucket(name=bucket).objects.all():
        obj_paths.append(os.path.join(obj.bucket, obj.key))
    return obj_paths


def parse_dicom_image_list(datadir, subdir, cache_path, verbose=False):
    """Get list of DICOMs from a local source and parse train/test and patientId."""

    # Load cached data
    if Path(cache_path).is_file():
        print('Using cached image list: %s' % cache_path)
        image_df = pd.read_csv(cache_path)
        return image_df
    else:
        print('Image list not found: %s' % cache_path)

    # TODO: SLOW. Not efficient. Find way not to do this in a loop.
    image_df = pd.DataFrame(columns=['path', 'subdir', 'patient_id'])
    for path in os.listdir(datadir):
        if path.endswith('.dcm'):
            if verbose:
                print(path)
            image_df = image_df.append({
                'path': datadir + '/' + path,
                'subdir': subdir,
                'patient_id': path.split('.')[0]},
                ignore_index=True)
    image_df.to_csv(cache_path)
    return image_df


def parse_s3_dicom_image_list(bucket, subdir='train', limit=None, verbose=False):
    """Get list of DICOMs from S3 bucket and parse train/test and patientId."""
    # TODO: SLOW. Not efficient. Find way not to do this in a loop.
    # TODO: Check if using `limit` is deterministic.
    print("Retrieving image list...")
    image_df = pd.DataFrame(columns=['path', 'subdir', 'patient_id'])
    s3_config = Config(connect_timeout=50, read_timeout=70)
    s3_resource = boto3.resource('s3', config=s3_config)  # higher-level OOO API
    for obj in s3_resource.Bucket(name=bucket).objects.all():
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


def read_s3_df(bucket, file_key):
    """Read CSV from S3 and return a pandas dataframe."""
    try:
        s3_config = Config(connect_timeout=50, read_timeout=70)
        s3_client = boto3.client('s3', config=s3_config)  # low-level functional API
        obj = s3_client.get_object(Bucket=bucket, Key=file_key)
        return pd.read_csv(obj['Body'])
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise


def get_s3_dcm(bucket, file_key):
    """Read DICOM from S3"""
    s3_config = Config(connect_timeout=50, read_timeout=70)
    s3_client = boto3.client('s3', config=s3_config)  # low-level functional API
    obj = s3_client.get_object(Bucket=bucket, Key=file_key)
    return pydicom.read_file(BytesIO(obj['Body'].read()))


def parse_training_labels(train_box_df, train_image_dirpath):
    """
    Reads and parses CSV file into a pandas dataframe.

    Example:
    {
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
    """
    # Define lambda to extract coords in list [x, y, width, height]
    def extract_box(row):
        return [row['x'], row['y'], row['width'], row['height']]

    parsed = {}
    for n, row in train_box_df.iterrows():
        # Initialize patient entry into parsed
        pid = row['patientId']
        if pid not in parsed:
            parsed[pid] = {
                'dicom': train_image_dirpath + '/%s.dcm' % pid,
                'label': row['Target'],
                'boxes': []}

        # Add box if opacity is present
        if parsed[pid]['label'] == 1:
            parsed[pid]['boxes'].append(extract_box(row))

    return parsed
