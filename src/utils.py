# -*- coding: utf-8 -*-
"""Utilities"""

import os
import sys
import random
import math
import numpy as np

try:
    from matplotlib import pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

try:
    script_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_path = "/projects/lungbox/src"
    sys.path.append(script_path)
sys.path.append(script_path + "/Mask_RCNN")

import Mask_RCNN.mrcnn.model as modellib
import Mask_RCNN.mrcnn.utils as modelutils

import ingest
from config import GlobalConfig


def split_dataset(ids, validation_split=0.2):
    """Split dataset for training and validation."""
    random.shuffle(ids)
    split_index = int((1 - validation_split) * len(ids))
    train_ids = ids[:split_index]
    valid_ids = ids[split_index:]
    print('Training count: %s' % len(train_ids))
    print('Validation instance count: %s' % len(valid_ids))
    return {'train_ids': train_ids,
            'valid_ids': valid_ids}


def split_dataset_by_class(annotation_dict, subset_size, validation_split):
    """Split dataset for training and validation, preserving outcome class distribution."""

    n_positive = sum(1 for v in annotation_dict.values() if v['label'] == 1)
    n_total = len(annotation_dict)
    p_positive = n_positive / n_total

    positive_ids = [k for k, v in annotation_dict.items() if v['label'] == 1]
    negative_ids = [k for k, v in annotation_dict.items() if v['label'] == 0]
    random.seed(GlobalConfig.get('RANDOM_SEED'))
    random.shuffle(positive_ids)
    random.shuffle(negative_ids)

    n_positive_subset = int(math.ceil(subset_size * p_positive))
    n_negative_subset = int(subset_size - n_positive_subset)

    # ceilings & floors to ensure that we get exactly subset_size*validation_split
    train_pos_index = int(math.ceil(n_positive_subset*(1-validation_split)))
    train_neg_index = int(math.floor(n_negative_subset*(1-validation_split)))

    train_ids = positive_ids[:train_pos_index] + negative_ids[:train_neg_index]
    valid_ids = positive_ids[train_pos_index:n_positive_subset] + \
        negative_ids[train_neg_index:n_negative_subset]

    print('Training count: %s' % len(train_ids))
    print('Validation instance count: %s' % len(valid_ids))
    return {'train_ids': train_ids,
            'valid_ids': valid_ids}


def find_latest_model(model_dir, prefix, inference_config, verbose=False):
    """Find path containing latest model."""

    dir_names = next(os.walk(model_dir))[1]
    key = inference_config.NAME.lower()
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)
    if not dir_names:
        import errno
        raise IOError(
            errno.ENOENT,
            "Could not find model directory under {}".format(model_dir))
    else:
        print(dir_names)

    fps = []
    for d in dir_names:
        dir_name = os.path.join(model_dir, d)
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith(prefix), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            if verbose:
                print('No weight files in {}'.format(dir_name))
        else:
            checkpoint = os.path.join(dir_name, checkpoints[-1])
            fps.append(checkpoint)
    if len(fps) == 0:
        print("No weight files found.")
        return None
    else:
        model_path = sorted(fps)[-1]
        print('Found model {}'.format(model_path))
        return model_path


def draw(parsed_df, patient_id):
    """
    Method to draw single patient with bounding box(es) if present.

    References:
    * Exploratory Data Analysis, by Peter Chang, MD.
        - Available at https://www.kaggle.com/peterchang77/exploratory-data-analysis/notebook
        - Retrieved 2018-09-19
    """
    d = ingest.get_s3_dcm(patient_id)
    im = d.pixel_array

    # Convert from single-channel grayscale to 3-channel RGB
    im = np.stack([im] * 3, axis=2)

    # Add boxes with random color if present
    for box in parsed_df[patient_id]['boxes']:
        rgb = np.floor(np.random.rand(3) * 256).astype('int')
        im = overlay_box(im=im, box=box, rgb=rgb, stroke=6)

    plt.imshow(im, cmap=plt.cm.gist_gray)
    plt.axis('off')


def overlay_box(im, box, rgb, stroke=1):
    """
    Method to overlay single box on image.

    References:
    * Exploratory Data Analysis, by Peter Chang, MD.
        - Available at https://www.kaggle.com/peterchang77/exploratory-data-analysis/notebook
        - Retrieved 2018-09-19
    """
    # Convert coordinates to integers
    box = [int(b) for b in box]

    # Extract coordinates
    y1, x1, height, width = box
    y2 = y1 + height
    x2 = x1 + width

    im[y1:y1 + stroke, x1:x2] = rgb
    im[y2:y2 + stroke, x1:x2] = rgb
    im[y1:y2, x1:x1 + stroke] = rgb
    im[y1:y2, x2:x2 + stroke] = rgb

    return im


def get_class_result(gt_class, pred_class):
    if 1 in gt_class and 1 in pred_class:
        return 'tp'
    elif 1 in gt_class and 1 not in pred_class:
        return 'fn'
    elif 1 not in gt_class and 1 in pred_class:
        return 'fp'
    else:
        return 'tn'


def compute_batch_metrics(model, dataset, inference_config, image_ids, verbose=False):

    all_results = dict()
    for image_id in image_ids:
        print("Processing image id: %s" % image_id)
        all_results[image_id] = dict()
        image, image_meta, gt_class_ids, gt_boxes, gt_masks =\
            modellib.load_image_gt(dataset, inference_config,
                                   image_id=image_id, use_mini_mask=False)
        all_results[image_id]['gt_class_ids'] = gt_class_ids
        all_results[image_id]['gt_boxes'] = gt_boxes
        all_results[image_id]['gt_masks'] = gt_masks

        # Run object detection
        r = model.detect([image], verbose=False)[0]
        class_result = get_class_result(gt_class=gt_class_ids, pred_class=r['class_ids'])
        all_results[image_id]['class_result'] = class_result
        all_results[image_id]['pred_class_ids'] = r['class_ids']
        all_results[image_id]['pred_boxes'] = r['rois']
        all_results[image_id]['pred_scores'] = r['scores']
        all_results[image_id]['pred_masks'] = r['masks']

        print("Ground Truth Classes: %s" % gt_class_ids)
        print("Predicted Classes: %s" % r['class_ids'])
        print("Predicted Scores: %s" % r['scores'])
        print("Class Result: %s" % class_result)

        if gt_masks.size > 0 and r['masks'].size > 0:
            gt_match, pred_match, overlaps = modelutils.compute_matches(
                gt_boxes=gt_boxes, gt_class_ids=gt_class_ids, gt_masks=gt_masks,
                pred_boxes=r['rois'], pred_class_ids=r['class_ids'],
                pred_scores=r['scores'], pred_masks=r['masks'],
                iou_threshold=0.5, score_threshold=0.0)
            all_results[image_id]['gt_match'] = gt_match
            all_results[image_id]['pred_match'] = pred_match
            all_results[image_id]['overlaps'] = overlaps
        elif gt_masks.size == 0 and r['masks'].size > 0:
            all_results[image_id]['overlaps'] = np.zeros(shape=(r['class_ids'].shape[0], 1))
        else:
            all_results[image_id]['overlaps'] = np.array([])

        print("Overlap: \n%s" % all_results[image_id]['overlaps'])
        print("\n")

    return all_results


def summarize_cm(class_counts):
    N = sum(class_counts.values())
    accuracy = (class_counts['tp'] + class_counts['tn']) / N
    recall = class_counts['tp'] / (class_counts['tp'] + class_counts['fn'])
    precision = class_counts['tp'] / (class_counts['tp'] + class_counts['fp'])
    f1 = 2 * ((precision * recall) / (precision + recall))
    print("Accuracy: %s" % accuracy)
    print("Recall: %s" % recall)
    print("Precision: %s" % precision)
    print("F1: %s" % f1)
    return {'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
            'f1': f1}
