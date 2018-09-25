import os
import sys
import numpy as np
import cv2

import ingest
script_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_path + "/Mask_RCNN")
import Mask_RCNN.mrcnn.model as modellib
from Mask_RCNN.mrcnn import utils
from Mask_RCNN.mrcnn import visualize
from Mask_RCNN.mrcnn.model import log


class DetectorDataset(utils.Dataset):
    """Dataset class for training pneumonia detection on the RSNA pneumonia dataset.

    References:
    * Intro to deep learning for medical imaging by MD.ai
        - https://github.com/mdai/ml-lessons/blob/master/lesson3-rsna-pneumonia-detection-mdai-client-lib.ipynb
        - Retrieved 2018-09-20
        - Licensed under the Apache License, Version 2.0
    """

    def __init__(self, patient_ids, annotation_dict, orig_height, orig_width, s3_bucket):
        super().__init__(self)
        self.s3_bucket = s3_bucket

        # Add classes
        self.add_class('pneumonia', 1, 'Lung Opacity')

        # Add images
        for i, patient_id in enumerate(patient_ids):
            self.add_image('pneumonia',
                           image_id=i,
                           path=annotation_dict[patient_id]['dicom_s3_key'],
                           annotations=annotation_dict[patient_id]['boxes'],
                           orig_height=orig_height,
                           orig_width=orig_width)

    def get_image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']

    def get_image_annotation(self, image_id):
        info = self.image_info[image_id]
        return info['annotations']

    def load_image(self, image_id):
        info = self.image_info[image_id]
        ds = ingest.get_s3_dcm(bucket=self.s3_bucket, file_key=info['path'])
        image = ds.pixel_array
        # If grayscale. Convert to RGB for consistency.
        if len(image.shape) != 3 or image.shape[2] != 3:
            image = np.stack((image,) * 3, -1)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        annotations = info['annotations']
        number_of_boxes = len(annotations)
        if number_of_boxes == 0:
            mask = np.zeros(
                (info['orig_height'], info['orig_width'], 1), dtype=np.uint8)
            class_ids = np.zeros((1,), dtype=np.int32)
        else:
            mask = np.zeros(
                (info['orig_height'], info['orig_width'], number_of_boxes), dtype=np.uint8)
            class_ids = np.zeros((number_of_boxes,), dtype=np.int32)
            for i, box in enumerate(annotations):
                x = int(box[0])
                y = int(box[1])
                w = int(box[2])
                h = int(box[3])
                mask_instance = mask[:, :, i].copy()
                cv2.rectangle(mask_instance, (x, y),
                              (x + w, y + h), 255, -1)
                mask[:, :, i] = mask_instance
                class_ids[i] = 1
        return mask.astype(np.bool), class_ids.astype(np.int32)
