# -*- coding: utf-8 -*-
"""
Unit Tests for low level training data ingestion.

Usage:
    python -m unittest tests/test_ingest.py
"""

import sys
import unittest
import pandas as pd

sys.path.append('src')
sys.path.append("src/Mask_RCNN")

from config import GlobalConfig
import ingest


class IngestTest(unittest.TestCase):

    @unittest.skip('Test depends on S3 access permissions.')
    def test_get_s3_dcm(self):
        bucket = GlobalConfig.get('S3_BUCKET_NAME')
        imgdir = GlobalConfig.get('S3_STAGE1_TRAIN_IMAGE_DIR')
        test_dcm_path = imgdir + '/0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm'
        ds = ingest.get_s3_dcm(bucket=bucket, file_key=test_dcm_path)
        self.assertEqual(ds.pixel_array.shape, (1024, 1024))

    def test_parse_training_labels(self):
        parsed = ingest.parse_training_labels(
            train_box_df=pd.read_csv(GlobalConfig.get('EXAMPLE_TRAIN_BOX_PATH')),
            train_image_dirpath=GlobalConfig.get('EXAMPLE_STAGE1_TRAIN_IMAGE_DIR'))

        # Negative Case
        self.assertEquals(
            parsed['0004cfab-14fd-4e49-80ba-63a80b6bddd6'],
            {'dicom': 'data/example/stage_1_train_images/0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm',
             'label': 0,
             'boxes': []})

        # Positive Case
        self.assertEquals(
            parsed['00436515-870c-4b36-a041-de91049b9ab4'],
            {'dicom': 'data/example/stage_1_train_images/00436515-870c-4b36-a041-de91049b9ab4.dcm',
             'label': 1,
             'boxes': [[264.0, 152.0, 213.0, 379.0], [562.0, 152.0, 256.0, 453.0]]})


if __name__ == '__main__':
    unittest.main()
