# lungbox: Localizing evidence of pneumonia in chest X-rays.

## Overview

Pneumonia kills millions worldwide, in both developing and developed countries. Chest X-rays are commonly used to diagnose pneumonia. However, limited access to radiologists, fast turn-around times required of radiologists, and challenges in interpreting chest X-rays limit the efficacy of these X-rays in triaging patients.

This project is a step toward making chest X-ray interpretation faster, by automating the detection and localization of visual evidence of pneumonia from chest X-rays.

https://docs.google.com/presentation/d/1_z2aPasOguu1kZyQtjgMY54DRGjCleBz5f6p3x5RmaU/edit?usp=sharing

# Setup

```
pip install -r requirements.txt
python setup.py install
```

# Usage

## Train

```
# Create virtual environment
conda create -n lungbox python=3.8
conda install tensorflow==1.10.0
python setup.py install

# Fetch Mask R-CNN 
git submodule update --init


# Create output directories
mkdir logs/
mkdir -p output/models

# Run training
python src/train.py \
  --data_source=example \
  --subset_size=2500 \
  --validation_split=0.2 \
  --epochs=100 \
  --learning_rate=0.002 \
  --model_dir='output/models' 2>&1 | tee logs/$(date +%Y%m%d%H%M)_train.log
```

### Inference

```
python src/infer.py \
      --image_path=data/example/stage_1_train_images/00a85be6-6eb0-421d-8acf-ff2dc0007e8a.dcm \
      --h5_path=output/models/pneumonia20181003T1228/mask_rcnn_pneumonia_0050.h5
      --model_dir='output/models'
```

## Results

### Example Predictions

<img src="https://github.com/victorwyee/lungbox/raw/master/notebooks/sample_images/gt-vs-pred_09326eb7-f4cb-4d8f-83c6-8ba7fb8b5ac7.png" width=400px><img src="https://github.com/victorwyee/lungbox/raw/master/notebooks/sample_images/gt-vs-pred_0b2057bc-4c6d-4c90-8975-94c02392e460.png" width=400px>
<img src="https://github.com/victorwyee/lungbox/raw/master/notebooks/sample_images/gt-vs-pred_3e07be0a-9693-4f9c-9295-9e72b3e2a872.png" width=400px><img src="https://github.com/victorwyee/lungbox/raw/master/notebooks/sample_images/gt-vs-pred_b17cb86b-e90c-4a31-926a-2e5b022c6275.png" width=400px>


# References
  * Mask R-CNN, retrieved 2018-09-19
    - https://github.com/matterport/Mask_RCNN
    - License: MIT License, retrieved 2018-09-20
  * Intro to deep learning for medical imaging by MD.ai
    - https://github.com/mdai/ml-lessons/blob/master/lesson3-rsna-pneumonia-detection-mdai-client-lib.ipynb
    - License: Apache License, Version 2.0


# License

MIT License

Copyright (c) 2018 Victor W. Yee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
