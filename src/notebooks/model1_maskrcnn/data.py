# -*- coding: utf-8 -*-
"""Initial model with Mask R-CNN: Data preparation."""

import random
import utils

import ingest
from model import DetectorDataset
from config import GlobalConfig


class TrainingData:

    DICOM_WIDTH = 1024
    DICOM_HEIGHT = 1024

    def __init__(self, subset_size, validation_split):
        random.seed(GlobalConfig.get('RANDOM_SEED'))
        self.train_box_df = self._retrieve_training_box_labels()
        self.annotation_dict = self._retrieve_annotation_dict(self.train_box_df)
        self.image_df = self._retrieve_dicom_image_list()

        # Split train/test sets
        patient_id_subset = sorted(list(
            self.image_df[self.image_df['subdir'] == 'train']['patient_id'])[:subset_size])
        id_split = utils.split_dataset(
            ids=patient_id_subset,
            validation_split=validation_split)
        self.patient_id_train = id_split['train_ids']
        self.patient_id_valid = id_split['valid_ids']

    def _retrieve_training_box_labels(self):
        train_box_df = ingest.read_s3_df(
            bucket=GlobalConfig.get('S3_BUCKET_NAME'),
            file_key=GlobalConfig.get('S3_TRAIN_BOX_KEY'))
        return train_box_df

    def _retrieve_annotation_dict(self, train_box_df):
        annotation_dict = ingest.parse_training_labels(
            train_box_df=train_box_df,
            train_image_dirpath=GlobalConfig.get('S3_STAGE1_TRAIN_IMAGE_DIR'))
        return annotation_dict

    def _retrieve_dicom_image_list(self):
        # Get list of images (SLOW before caching)
        image_df = ingest.parse_dicom_image_list(
            bucket=GlobalConfig.get('S3_BUCKET_NAME'))
        return image_df

    def get_train_ids(self):
        return self.patient_id_train

    def get_valid_ids(self):
        return self.patient_id_valid

    def get_dataset_train(self):
        dataset_train = DetectorDataset(
            patient_ids=self.patient_id_train,
            annotation_dict=self.annotation_dict,
            orig_height=self.DICOM_HEIGHT,
            orig_width=self.DICOM_WIDTH,
            s3_bucket=GlobalConfig.get('S3_BUCKET_NAME'))
        dataset_train.prepare()
        assert len(dataset_train.image_ids) == len(self.patient_id_train)
        return dataset_train

    def get_dataset_valid(self):
        dataset_valid = DetectorDataset(
            patient_ids=self.patient_id_valid,
            annotation_dict=self.annotation_dict,
            orig_height=self.DICOM_HEIGHT,
            orig_width=self.DICOM_WIDTH,
            s3_bucket=GlobalConfig.get('S3_BUCKET_NAME'))
        dataset_valid.prepare()
        assert len(dataset_valid.image_ids) == len(self.patient_id_valid)
        return dataset_valid
