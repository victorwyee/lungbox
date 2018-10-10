# -*- coding: utf-8 -*-

import cv2
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
from skimage.measure import find_contours
import colorsys

try:
    script_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_path = "/projects/lungbox/src"
    sys.path.append(script_path)
sys.path.append(script_path + "/Mask_RCNN")

import Mask_RCNN.mrcnn.utils as modelutils


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def get_colors_for_class_ids(class_ids):
    """Set color for class."""
    colors = []
    for class_id in class_ids:
        if class_id == 1:
            colors.append((.941, .204, .204))
    return colors


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
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
    colors = colors or random_colors(N)

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
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            x = random.randint(x1, (x1 + x2) // 2)
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Maska
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


def show_random_predictions(dataset, inference_config, model, number_of_images=3, verbose=1):
    """Show some examples of ground truth vs. predictions on the validation set."""
    fig = plt.figure(figsize=(10, 20))
    num_imgs = 3
    for i in range(num_imgs):
        image_id = random.choice(dataset.image_ids)
        print(image_id)
        original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(
                dataset=dataset, config=inference_config, image_id=image_id,
                augment=False, augmentation=None,use_mini_mask=False)
        plt.subplot(num_imgs, 2, 2 * i + 1)
        visualize.display_instances(
            image=original_image,
            boxes=gt_bbox, masks=gt_mask,
            class_ids=gt_class_id, class_names=dataset.class_names,
            colors=get_colors_for_class_ids(gt_class_id), ax=fig.axes[-1])
        plt.title('Ground Truth')
        plt.subplot(num_imgs, 2, 2 * i + 2)

        results = model.detect([original_image], verbose=verbose)
        r = results[0]
        visualize.display_instances(
            image=original_image,
            boxes=r['rois'], masks=r['masks'],
            class_ids=r['class_ids'], class_names=dataset.class_names,
            scores=r['scores'],
            colors=get_colors_for_class_ids(r['class_ids']), ax=fig.axes[-1])
        plt.title('Prediction')


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image."""
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def display_box_differences(image,
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
    colors = [(0, 1, 0, .8)] * len(gt_match) + [(1, 0, 0, 1)] * len(pred_match)
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
    title = title or "Ground Truth and Detections\n GT=green, pred=red, captions: score/IoU"
    # Display
    display_instances(
        image,
        boxes, masks, class_ids,
        class_names, scores, ax=ax,
        show_bbox=show_box, show_mask=show_mask,
        colors=colors, captions=captions,
        title=title)


def display_differences_by_id(dataset, inference_config, image_id,
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
                        show_mask=show_mask, show_box=show_box,
                        iou_threshold=iou_threshold, score_threshold=score_threshold)
