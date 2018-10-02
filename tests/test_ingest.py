# -*- coding: utf-8 -*-
"""Unit Tests for Data Ingestion"""


import sys
import unittest

sys.path.append('../src')
sys.path.append("../src/Mask_RCNN")

from config import GlobalConfig
import ingest


class IngestTest(unittest.TestCase):

    def test_get_s3_dcm(self):
        bucket = GlobalConfig.get('S3_BUCKET_NAME')
        imgdir = GlobalConfig.get('S3_STAGE1_TRAIN_IMAGE_DIR')
        test_dcm_path = imgdir + '/0004cfab-14fd-4e49-80ba-63a80b6bddd6.dcm'
        ds = ingest.get_s3_dcm(bucket=bucket, file_key=test_dcm_path)
        self.assertEqual(ds.pixel_array.shape, (1024, 1024))


if __name__ == '__main__':
    unittest.main()
