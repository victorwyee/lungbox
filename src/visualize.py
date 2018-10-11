# -*- coding: utf-8 -*-

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon
from skimage.measure import find_contours

try:
    script_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_path = "/projects/lungbox/src"
    sys.path.append(script_path)
sys.path.append(script_path + "/Mask_RCNN")

import Mask_RCNN.mrcnn.model as modellib
import Mask_RCNN.mrcnn.utils as modelutils
import Mask_RCNN.mrcnn.visualize as visualize


def get_colors_for_class_ids(class_ids):
    """Set color for class."""
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors


def display_boxes(image, boxes, title="",
                  figsize=(10, 10), ax=None, colors=None):

    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = [(0, 1, 0, .8)] * len(boxes)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    # Copy image
    masked_image = image.astype(np.uint32).copy()

    # Adjust for grayscale images
    if masked_image.ndim == 3 and masked_image.shape[2] == 1:
        grayscale = True

    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue

        y1, x1, y2, x2 = boxes[i]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

    if grayscale:
        masked_image = masked_image[:, :, 0]
        ax.imshow(masked_image.astype(np.uint8), cmap='gray')
    else:
        ax.imshow(masked_image.astype(np.uint8))

    if auto_show:
        plt.show()


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=False, show_bbox=True,
                      colors=None, captions=None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    if masked_image.ndim == 3 and masked_image.shape[2] == 1:
        grayscale = True

    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2,
                alpha=0.7, linestyle='dashed',
                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Masks
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

    if grayscale:
        masked_image = masked_image[:, :, 0]
        ax.imshow(masked_image.astype(np.uint8), cmap='gray')
    else:
        ax.imshow(masked_image.astype(np.uint8))

    if auto_show:
        plt.show()


def show_randoms(model, dataset, inference_config, n_images=3):
    fig = plt.figure(figsize=(10, 20))
    for i in range(n_images):

        image_id = random.choice(dataset.image_ids)
        print('Image ID: {}'.format(image_id))
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset, inference_config,
                                   image_id, use_mini_mask=False)

        plt.subplot(n_images, 2, 2 * i + 1)
        display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
                          dataset.class_names,
                          show_mask=False,
                          colors=get_colors_for_class_ids(gt_class_id),
                          ax=fig.axes[-1])
        plt.title('Ground Truth')

        plt.subplot(n_images, 2, 2 * i + 2)

        # turn off verbose if you don't want debug messages
        results = model.detect([original_image], verbose=1)
        r = results[0]
        display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                          dataset.class_names, r['scores'],
                          show_mask=False,
                          colors=get_colors_for_class_ids(r['class_ids']),
                          ax=fig.axes[-1])
        plt.title('Prediction')


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image."""
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_differences(image,
                        gt_box, gt_class_id, gt_mask,
                        pred_box, pred_class_id, pred_score, pred_mask,
                        class_names, title="", ax=None,
                        show_mask=False, show_box=True,
                        iou_threshold=0.5, score_threshold=0.5):
    """Display ground truth and prediction instances on the same image."""

    # Match predictions to ground truth
    gt_match, pred_match, overlaps = modelutils.compute_matches(
        gt_box, gt_class_id, gt_mask,
        pred_box, pred_class_id, pred_score, pred_mask,
        iou_threshold=iou_threshold, score_threshold=score_threshold)

    # Ground truth = green. Predictions = red
    colors = [(0, 1, 0, .8)] * len(gt_match)\
        + [(1, 0, 0, 1)] * len(pred_match)

    # Concatenate GT and predictions
    class_ids = np.concatenate([gt_class_id, pred_class_id])
    scores = np.concatenate([np.zeros([len(gt_match)]), pred_score])
    boxes = np.concatenate([gt_box, pred_box])
    masks = np.concatenate([gt_mask, pred_mask], axis=-1)

    # Captions per instance show score/IoU
    captions = ["" for m in gt_match] + ["{:.2f} / {:.2f}".format(
        pred_score[i],
        (overlaps[i, int(pred_match[i])]
            if pred_match[i] > -1 else overlaps[i].max()))
        for i in range(len(pred_match))]

    # Set title if not provided
    title = title or "Ground Truth and Detections\n GT: Green, Prediction: Red, Captions: score/IoU"

    # Display
    display_instances(
        image,
        boxes, masks, class_ids,
        class_names, scores, ax=ax,
        show_bbox=show_box, show_mask=show_mask,
        colors=colors, captions=captions,
        title=title)


def display_differences_by_id(model, dataset, inference_config, image_id,
                              title="", ax=None, show_mask=False, show_box=True,
                              iou_threshold=0.5, score_threshold=0.5):
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, inference_config,
                               image_id, use_mini_mask=False)
    r = model.detect([original_image], verbose=True)[0]
    display_differences(original_image,
                        gt_bbox, gt_class_id, gt_mask,
                        r['rois'], r['class_ids'], r['scores'], r['masks'],
                        dataset.class_names, title=title, ax=ax,
                        show_mask=False, show_box=show_box,
                        iou_threshold=iou_threshold, score_threshold=score_threshold)
