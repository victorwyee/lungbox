# -*- coding: utf-8 -*-
"""Unit Tests for the TrainingData class"""

import os
import sys
import unittest

# os.chdir('/projects/lungbox')
os.chdir('..')  # Move to project directory
sys.path.append('src')
sys.path.append("src/Mask_RCNN")

from data import TrainingData


class TrainingDataTest(unittest.TestCase):

    # TODO: The list of DICOM images gets cached after initial load.
    # Currently, this test class will only test the non-cached branch on initial
    # load and the cached branch in subsequent test runs.

    # TODO: Cleaner separation of dev/prod/test environment variables.

    EXAMPLE_TRAINING_PATIENT_IDS = [
     '00c0b293-48e7-4e16-ac76-9269ba535a62',
     '00704310-78a8-4b38-8475-49f4573b2dbb',
     '00aecb01-a116-45a2-956c-08d2fa55433f',
     '00f08de1-517e-4652-a04f-d1dc9ee48593',
     '00a85be6-6eb0-421d-8acf-ff2dc0007e8a',
     '009482dc-3db5-48d4-8580-5c89c4f01334',
     '009eb222-eabc-4150-8121-d5a6d06b8ebf',
     '0004cfab-14fd-4e49-80ba-63a80b6bddd6',
     '00569f44-917d-4c86-a842-81832af98c30',
     '008c19e8-a820-403a-930a-bc74a4053664',
     '006cec2e-6ce2-4549-bffa-eadfcd1e9970',
     '00313ee0-9eaa-42f4-b0ab-c148ed3241cd']

    EXAMPLE_VALIDATION_PATIENT_IDS = [
     '00436515-870c-4b36-a041-de91049b9ab4',
     '00322d4d-1c29-4943-afc9-b6754be640eb',
     '00d7c36e-3cdf-4df6-ac03-6c30cdc8e85b',
     '003d8fa0-6bf1-40ed-b54c-ac657f8495c5']

    def test_local_train_data_init(self):
        local_train_data = TrainingData(
            subset_size=16,
            validation_split=0.2,
            data_source='example',
            subdir='train')
        self.assertEqual(local_train_data.subset_size, 16)
        self.assertEqual(local_train_data.validation_split, 0.2)
        self.assertEqual(local_train_data.data_source, 'example')
        self.assertEqual(local_train_data.subdir, 'train')

        self.assertEqual(local_train_data.train_box_df.shape, (21, 6))
        self.assertEqual(len(local_train_data.annotation_dict), 16)
        self.assertEqual(local_train_data.image_df.shape, (16, 4))

        self.assertEqual(
            sum(x['label'] for x in local_train_data.annotation_dict.values()),
            5, "Expected 5 positive cases.")
        self.assertEqual(
            sum(1 for id in local_train_data.patient_id_train
                if len(local_train_data.annotation_dict[id]['boxes']) != 0),
            4, "Expected 4 positive cases in training set.")
        self.assertEqual(
            sum(1 for id in local_train_data.patient_id_valid
                if len(local_train_data.annotation_dict[id]['boxes']) != 0),
            1, "Expected 1 positive cases in validation set.")

        self.assertEqual(
            local_train_data.patient_id_train, self.EXAMPLE_TRAINING_PATIENT_IDS)
        self.assertEqual(
            local_train_data.patient_id_valid, self.EXAMPLE_VALIDATION_PATIENT_IDS)

    def test_get_dataset_train(self):
        local_train_data = TrainingData(
            subset_size=16,
            validation_split=0.2,
            data_source='example',
            subdir='train')
        dataset_train = local_train_data.get_dataset_train()

        # Order should be the retained (even though it's not important)
        self.assertEqual(dataset_train.patient_ids[0], local_train_data.patient_id_train)
        self.assertEqual(dataset_train.annotation_dict, local_train_data.annotation_dict)
        self.assertEqual(dataset_train.orig_height, 1024)
        self.assertEqual(dataset_train.orig_width, 1024)
        self.assertEqual(dataset_train.data_source, 'example')
        self.assertEqual(dataset_train.s3_bucket, None)


if __name__ == '__main__':
    unittest.main()
