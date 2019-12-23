"""Based on https://github.com/keras-team/keras/blob/master/examples/deep_dream.py"""

import argparse
from typing import Callable, Dict, List, Tuple

import keras
import numpy as np
import scipy
import tensorflow as tf
from keras import backend as K
from keras.applications import inception_v3
from keras.preprocessing.image import load_img, save_img, img_to_array


def deep_dream(base_image_path,
               result_prefix,
               step: float = 0.01,  # Gradient ascent step size
               num_octave: int = 3,  # Number of scales at which to run gradient ascent
               octave_scale: float = 1.4,  # Size ratio between scales
               iterations: int = 20,  # Number of ascent steps per scale
               max_loss: float = 10.):  # Max allowed loss

    print(f"Initiating deep dream with the following parameters:\n"
          f"step={step}\n"
          f"num_octave={num_octave}\n"
          f"octave_scale={octave_scale}\n"
          f"iterations={iterations}\n"
          f"max_loss={max_loss}\n")

    # These are the names of the layers
    # for which we try to maximize activation,
    # as well as their weight in the final loss
    # we try to maximize.
    # You can tweak these setting to obtain new visual effects.
    settings: Dict[str, Dict[str, float]] = {
        "features": {
            "mixed2": 0.2,
            "mixed3": 0.5,
            "mixed4": 2.,
            "mixed5": 1.5,
        },
    }

    def preprocess_image(image_path: str):
        # Util function to open, resize and format pictures
        # into appropriate tensors.
        _img: np.ndarray = load_img(image_path)
        _img = img_to_array(_img)
        _img = np.expand_dims(_img, axis=0)
        _img = inception_v3.preprocess_input(_img)
        return _img

    def deprocess_image(_x: np.ndarray):
        # Util function to convert a tensor into a valid image.
        if K.image_data_format() == "channels_first":
            _x = _x.reshape((3, _x.shape[2], _x.shape[3]))
            _x = _x.transpose((1, 2, 0))
        else:
            _x = _x.reshape((_x.shape[1], _x.shape[2], 3))
        _x /= 2.
        _x += 0.5
        _x *= 255.
        _x = np.clip(_x, 0, 255).astype("uint8")
        return _x

    K.set_learning_phase(0)

    # Build the InceptionV3 network with our placeholder.
    # The model will be loaded with pre-trained ImageNet weights.
    model: keras.Model = inception_v3.InceptionV3(weights="imagenet",
                                                  include_top=False)
    dream: tf.Tensor = model.input
    print("Model loaded.")

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict: Dict[str, keras.layers.Layer] = dict([(layer.name, layer) for layer in model.layers])

    # Define the loss.
    loss: tf.Tensor = K.variable(0.)
    for layer_name in settings["features"]:
        # Add the L2 norm of the features of a layer to the loss.
        if layer_name not in layer_dict:
            raise ValueError("Layer " + layer_name + " not found in model.")
        coeff: float = settings["features"][layer_name]
        x: tf.Tensor = layer_dict[layer_name].output
        # We avoid border artifacts by only involving non-border pixels in the loss.
        scaling: tf.Tensor = K.prod(K.cast(K.shape(x), "float32"))
        if K.image_data_format() == "channels_first":
            loss = loss + coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
        else:
            loss = loss + coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

    # Compute the gradients of the dream wrt the loss.
    grads: tf.Tensor = K.gradients(loss, dream)[0]
    # Normalize gradients.
    grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())

    # Set up function to retrieve the value
    # of the loss and gradients given an input image.
    outputs: List = [loss, grads]
    fetch_loss_and_grads: Callable = K.function([dream], outputs)

    def eval_loss_and_grads(_x: np.ndarray) -> Tuple[float, np.ndarray]:
        outs: List = fetch_loss_and_grads([_x])
        loss_value: float = outs[0]
        grad_values: np.ndarray = outs[1]
        return loss_value, grad_values

    def resize_img(_img: np.ndarray, size):
        _img = np.copy(_img)
        if K.image_data_format() == "channels_first":
            factors = (1, 1,
                       float(size[0]) / _img.shape[2],
                       float(size[1]) / _img.shape[3])
        else:
            factors: Tuple = (1,
                              float(size[0]) / _img.shape[1],
                              float(size[1]) / _img.shape[2],
                              1)
        return scipy.ndimage.zoom(_img, factors, order=1)

    def gradient_ascent(_x: np.ndarray, _iterations: int, _step: float, _max_loss: float = None) -> np.ndarray:
        for _i in range(_iterations):
            loss_value, grad_values = eval_loss_and_grads(_x)
            if _max_loss is not None and loss_value > _max_loss:
                break
            print("..Loss value at", _i, ":", loss_value)
            _x += _step * grad_values
        return _x

    """Process:
    
    - Load the original image.
    - Define a number of processing scales (i.e. image shapes),
        from smallest to largest.
    - Resize the original image to the smallest scale.
    - For every scale, starting with the smallest (i.e. current one):
        - Run gradient ascent
        - Upscale image to the next scale
        - Reinject the detail that was lost at upscaling time
    - Stop when we are back to the original size.
    
    To obtain the detail lost during upscaling, we simply
    take the original image, shrink it down, upscale it,
    and compare the result to the (resized) original image.
    """

    img = preprocess_image(base_image_path)
    if K.image_data_format() == "channels_first":
        original_shape: Tuple = img.shape[2:]
    else:
        original_shape = img.shape[1:3]
    successive_shapes: List[Tuple] = [original_shape]
    for i in range(1, num_octave):
        shape: Tuple = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)
    successive_shapes = successive_shapes[::-1]
    original_img: np.ndarray = np.copy(img)
    shrunk_original_img: np.ndarray = resize_img(img, successive_shapes[0])

    for shape in successive_shapes:
        print("Processing image shape", shape)
        img = resize_img(img, shape)
        img = gradient_ascent(img,
                              _iterations=iterations,
                              _step=step,
                              _max_loss=max_loss)
        upscaled_shrunk_original_img: np.ndarray = resize_img(shrunk_original_img, shape)
        same_size_original: np.ndarray = resize_img(original_img, shape)
        lost_detail: np.ndarray = same_size_original - upscaled_shrunk_original_img

        img += lost_detail
        shrunk_original_img = resize_img(original_img, shape)

    save_img(result_prefix + ".png", deprocess_image(np.copy(img)))


if __name__ == "__main__":

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Deep Dreams with Keras.")
    parser.add_argument("base_image_path", metavar="base", type=str, help="Path to the image to transform.")
    parser.add_argument("result_prefix", metavar="res_prefix", type=str, help="Prefix for the saved results.")

    parser.add_argument("--step", type=float, default=0.01, help="Gradient ascent step size.")
    parser.add_argument("--num_octave", type=int, default=3, help="Number of scales at which to run gradient ascent.")
    parser.add_argument("--octave_scale", type=float, default=1.4, help="Size ratio between scales.")
    parser.add_argument("--iterations", type=int, default=20, help="Number of ascent steps per scale.")
    parser.add_argument("--max_loss", type=float, default=10, help="Max allowed loss.")

    args: argparse.Namespace = parser.parse_args()
    _base_image_path: str = args.base_image_path
    _result_prefix: str = args.result_prefix
    _step: float = args.step
    _num_octave: int = args.num_octave
    _octave_scale: float = args.octave_scale
    _iterations: int = args.iterations
    _max_loss: float = args.max_loss

    deep_dream(_base_image_path,
               _result_prefix,
               step=_step,
               num_octave=_num_octave,
               octave_scale=_octave_scale,
               iterations=_iterations,
               max_loss=_max_loss)
