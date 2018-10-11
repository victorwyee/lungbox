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
python src/train.py \
  --data_source=example \
  --subset_size=2500 \
  --validation_split=0.2 \
  --epochs=100 \
  --learning_rate=0.002 \
  --model_dir='/projects/lungbox/models' 2>&1 | tee logs/$(date +%Y%m%d%H%M)_train.log
```

### Inference

```
python src/infer.py \
      --image_path=/projects/lungbox/data/example/stage_1_train_images/00a85be6-6eb0-421d-8acf-ff2dc0007e8a.dcm \
      --h5_path=/projects/lungbox/models/pneumonia20181003T1228/mask_rcnn_pneumonia_0050.h5
      --model_dir='/projects/lungbox/models'
```

## Results



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
