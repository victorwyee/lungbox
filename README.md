# lungbox: Localizing evidence of pneumonia in chest X-rays.

A project to localize areas of lung opacities, to predict pneumonia, from chest X-rays.

# Setup

```
pip install -r requirements.txt
python setup.py install
```

# Usage

## Train

```
# max training set size = 25684
python src/notebooks/model1_maskrcnn/train.py \
    --subset_size=1000 \
    --validation_split=0.2 \
    --epochs=1 2>&1 | tee logs/$(date +%Y%m%d%H%M)_train.log
```

At this time, all data is hosted on S3. Example data will be added shortly.

## Inference

In progress.

# Example Results

In progress.

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
