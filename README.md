# lungbox: Localizing evidence of pneumonia in chest X-rays.

# Setup

```
pip install -r requirements.txt
python setup.py install
```

## Usage

### Train

```
# max training set size = 25684
python src/notebooks/model1_maskrcnn/train.py \
    --subset_size=1000 \
    --validation_split=0.2 \
    --epochs=1 2>&1 | tee logs/$(date +%Y%m%d%H%M)_train.log
```

At this time, all data is hosted on S3. Example data will be added shortly.

### Inference

In progress.

## Results

In progress.

## References
  * Mask R-CNN, retrieved 2018-09-19
    - https://github.com/matterport/Mask_RCNN
    - License: MIT License, retrieved 2018-09-20
  * Intro to deep learning for medical imaging by MD.ai
    - https://github.com/mdai/ml-lessons/blob/master/lesson3-rsna-pneumonia-detection-mdai-client-lib.ipynb
    - License: Apache License, Version 2.0


# License

Apache License 2.0, January 2004
Copyright (c) 2018 Victor Yee
