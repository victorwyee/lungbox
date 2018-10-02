# -*- coding: utf-8 -*-
"""Initial model with Mask R-CNN: Data preparation."""

import os
import sys
import random
import pandas as pd

try:
    script_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_path = "/projects/lungbox/src"
    sys.path.append(script_path)
sys.path.append(script_path + "/Mask_RCNN")

import utils
import ingest
from model import DetectorDataset
from config import GlobalConfig


class TrainingData:

    DICOM_WIDTH = 1024
    DICOM_HEIGHT = 1024

    def __init__(self, subset_size, validation_split, data_source='example', subdir='train'):
        random.seed(GlobalConfig.get('RANDOM_SEED'))

        self.subset_size = subset_size
        self.validation_split = validation_split
        self.data_source = data_source
        self.subdir = subdir
        assert self.data_source in ["local", "s3", "example"]
        assert self.subdir in ["train", "test"]

        self.train_box_df = self._retrieve_training_box_labels()
        self.annotation_dict = self._retrieve_annotation_dict(self.train_box_df)
        self.image_df = self._retrieve_dicom_image_list()

        # Split train/test sets, preserving outcome class distribution
        id_split = utils.split_dataset_by_class(
            annotation_dict=self.annotation_dict,
            subset_size=subset_size,
            validation_split=validation_split)
        self.patient_id_train = id_split['train_ids']
        self.patient_id_valid = id_split['valid_ids']

    def _retrieve_dicom_image_list(self):
        if self.data_source == 'local':
            cache_path = GlobalConfig.get('LOCAL_DICOM_IMAGE_LIST_PATH')
            if self.subdir == 'train':
                datadir = GlobalConfig.get('S3_STAGE1_TRAIN_IMAGE_DIR')
            elif self.subdir == 'test':
                datadir = GlobalConfig.get('S3_STAGE1_TEST_IMAGE_DIR')
        elif self.data_source == 'example':
            cache_path = GlobalConfig.get('EXAMPLE_DICOM_IMAGE_LIST_PATH')
            datadir = GlobalConfig.get('EXAMPLE_STAGE1_TRAIN_IMAGE_DIR')
        elif self.data_source == 's3':
            print('S3 data source selected. Caching functionality not yet available.')
            return ingest.parse_s3_dicom_image_list(
                bucket=GlobalConfig.get('S3_BUCKET_NAME'),
                subdir=self.subdir)

        image_df = ingest.parse_dicom_image_list(
            datadir=datadir, subdir=self.subdir,
            cache_path=cache_path, verbose=False)
        return image_df

    def _retrieve_training_box_labels(self):
        if self.data_source == 'local':
            train_box_df = pd.read_csv(GlobalConfig.get('LOCAL_TRAIN_BOX_PATH'))
        elif self.data_source == 'example':
            train_box_df = pd.read_csv(GlobalConfig.get('EXAMPLE_TRAIN_BOX_PATH'))
        elif self.data_source == 's3':
            train_box_df = ingest.read_s3_df(
                bucket=GlobalConfig.get('S3_BUCKET_NAME'),
                file_key=GlobalConfig.get('S3_TRAIN_BOX_KEY'))
        return train_box_df

    def _retrieve_annotation_dict(self, train_box_df):
        if self.data_source == 'local':
            dirpath = GlobalConfig.get('LOCAL_STAGE1_TRAIN_IMAGE_DIR')
        if self.data_source == 'example':
            dirpath = GlobalConfig.get('EXAMPLE_STAGE1_TRAIN_IMAGE_DIR')
        elif self.data_source == 's3':
            dirpath = GlobalConfig.get('S3_STAGE1_TRAIN_IMAGE_DIR')
        annotation_dict = ingest.parse_training_labels(
            train_box_df=train_box_df,
            train_image_dirpath=dirpath)
        return annotation_dict

    def get_dataset_train(self):
        dataset_train = DetectorDataset(
            patient_ids=self.patient_id_train,
            annotation_dict=self.annotation_dict,
            orig_height=self.DICOM_HEIGHT,
            orig_width=self.DICOM_WIDTH,
            data_source=self.data_source,
            s3_bucket=(GlobalConfig.get('S3_BUCKET_NAME') if self.data_source == 's3' else None))
        dataset_train.prepare()
        assert len(dataset_train.image_ids) == len(self.patient_id_train)
        return dataset_train

    def get_dataset_valid(self):
        dataset_valid = DetectorDataset(
            patient_ids=self.patient_id_valid,
            annotation_dict=self.annotation_dict,
            orig_height=self.DICOM_HEIGHT,
            orig_width=self.DICOM_WIDTH,
            data_source=self.data_source,
            s3_bucket=(GlobalConfig.get('S3_BUCKET_NAME') if self.data_source == 's3' else None))
        dataset_valid.prepare()
        assert len(dataset_valid.image_ids) == len(self.patient_id_valid)
        return dataset_valid
