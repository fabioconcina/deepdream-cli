# DeepDream simple CLI
Based on https://github.com/keras-team/keras/blob/master/examples/deep_dream.py  
Original script has been refactored to allow for easier hyperparameters tuning from command line.

### Model

The model used is the **Inception V3** model for Keras
(https://github.com/keras-team/keras-applications/blob/master/keras_applications/inception_v3.py),
with weights pre-trained on ImageNet.

### Installation

Create a virtual enviroment, activate it and run

```bash
pip install -r requirements.txt
```

### Usage

```
usage: deep_dream.py [-h] [--step STEP] [--num_octave NUM_OCTAVE]
                     [--octave_scale OCTAVE_SCALE] [--iterations ITERATIONS]
                     [--max_loss MAX_LOSS] [--mixed2_weight MIXED2_WEIGHT]
                     [--mixed3_weight MIXED3_WEIGHT]
                     [--mixed4_weight MIXED4_WEIGHT]
                     [--mixed5_weight MIXED5_WEIGHT]
                     base_image_path result_prefix

Deep Dreams with Keras.

positional arguments:
  base_image_path       Path to the image to transform.
  result_prefix         Prefix for the saved results.

optional arguments:
  -h, --help            show this help message and exit
  --step STEP           Gradient ascent step size.
  --num_octave NUM_OCTAVE
                        Number of scales at which to run gradient ascent.
  --octave_scale OCTAVE_SCALE
                        Size ratio between scales.
  --iterations ITERATIONS
                        Number of ascent steps per scale.
  --max_loss MAX_LOSS   Max allowed loss.
  --mixed2_weight MIXED2_WEIGHT
                        Mixed layer 2 loss weight.
  --mixed3_weight MIXED3_WEIGHT
                        Mixed layer 3 loss weight.
  --mixed4_weight MIXED4_WEIGHT
                        Mixed layer 4 loss weight.
  --mixed5_weight MIXED5_WEIGHT
                        Mixed layer 5 loss weight.
```

For example, using default parameters:
```bash
python deep_dream.py img/test.jpg img/dream
```

### Optional parameters
- `step`: Gradient ascent step size (default 0.01).
- `num_octave`: Number of scales at which to run gradient ascent (default 3).
- `octave_scale`: Size ratio between scales (default 1.4).
- `iterations`: Number of ascent steps per scale (default 20).
- `max_loss`: Max allowed loss (default 10.0).
- `mixed2_weight`: Mixed layer 2 loss weight (default 0.2).
- `mixed3_weight`: Mixed layer 3 loss weight (default 0.5).
- `mixed4_weight`: Mixed layer 4 loss weight (default 2.0).
- `mixed5_weight`: Mixed layer 5 loss weight (default 1.5).

To change an hyperparameter, simply pass it as optional parameter
from command line:

```bash
python deep_dream.py img/test.jpg img/dream --num_octave 5
```
