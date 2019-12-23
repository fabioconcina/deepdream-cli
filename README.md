# DeepDream simple CLI
Based on https://github.com/keras-team/keras/blob/master/examples/deep_dream.py  
Original script has been refactored to allow for easier hyperparameters tuning from command line.

### Usage
```bash
python deep_dream.py base_image_path result_prefix
```

For example:
```bash
python deep_dream.py img/test.jpg img/dream
```

### Optional parameters
- `step`: float = 0.01,  # Gradient ascent step size
- `num_octave`: int = 3,  # Number of scales at which to run gradient ascent
- `octave_scale`: float = 1.4,  # Size ratio between scales
- `iterations`: int = 20,  # Number of ascent steps per scale
- `max_loss`: float = 10  # Max allowed loss

To change an hyperparameter, simply pass it as option parameter
from command line:

```bash
python deep_dream.py img/test.jpg img/dream --num_octave 5
```

### Installation

Create a virtual enviroment, activate it and run

```bash
pip install -r requirements.txt
```
