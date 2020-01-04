"""Based on https://github.com/keras-team/keras/blob/master/examples/deep_dream.py"""

import argparse
import logging
from typing import Callable, Dict, List, Optional, Tuple

import keras
import numpy as np
import scipy
import tensorflow as tf
from keras import backend as K  # noqa: N812
from keras.applications import inception_v3
from keras.preprocessing.image import img_to_array, load_img, save_img

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class DeepDream(object):

    # default parameters
    STEP: float = 0.01  # Gradient ascent step size
    NUM_OCTAVE: int = 3  # Number of scales at which to run gradient ascent
    OCTAVE_SCALE: float = 1.4  # Size ratio between scales
    ITERATIONS: int = 20  # Number of ascent steps per scale
    MAX_LOSS: float = 10.0  # Max allowed loss
    MIXED2_WEIGHT: float = 0.2  # Mixed layer 2 loss weight
    MIXED3_WEIGHT: float = 0.5  # Mixed layer 3 loss weight
    MIXED4_WEIGHT: float = 2.0  # Mixed layer 4 loss weight
    MIXED5_WEIGHT: float = 1.5   # Mixed layer 5 loss weight

    def __init__(self, base_image_path: str,
                 result_prefix: str,
                 step: float = STEP,
                 num_octave: int = NUM_OCTAVE,
                 octave_scale: float = OCTAVE_SCALE,
                 iterations: int = ITERATIONS,
                 max_loss: float = MAX_LOSS,
                 mixed2_weight: float = MIXED2_WEIGHT,
                 mixed3_weight: float = MIXED3_WEIGHT,
                 mixed4_weight: float = MIXED4_WEIGHT,
                 mixed5_weight: float = MIXED5_WEIGHT):
        self.base_image_path = base_image_path
        self.result_prefix = result_prefix
        self.step = step
        self.num_octave = num_octave
        self.octave_scale = octave_scale
        self.iterations = iterations
        self.max_loss = max_loss
        self.mixed2_weight = mixed2_weight
        self.mixed3_weight = mixed3_weight
        self.mixed4_weight = mixed4_weight
        self.mixed5_weight = mixed5_weight

    @staticmethod
    def from_dict(d: Dict) -> "DeepDream":
        """Creates an instance of DeepDream from dictionary.
        Optional parameters not found in the dictionary are defaulted to class defaults."""

        base_image_path = d.get("base_image_path")
        if base_image_path is None:
            raise ValueError("base_image_path param is required.")
        result_prefix = d.get("result_prefix")
        if result_prefix is None:
            raise ValueError("result_prefix param is required.")

        # optional params
        step = d.get("step", DeepDream.STEP)
        num_octave = d.get("num_octave", DeepDream.NUM_OCTAVE)
        octave_scale = d.get("octave_scale", DeepDream.OCTAVE_SCALE)
        iterations = d.get("iterations", DeepDream.ITERATIONS)
        max_loss = d.get("max_loss", DeepDream.MAX_LOSS)
        mixed2_weight = d.get("mixed2_weight", DeepDream.MIXED2_WEIGHT)
        mixed3_weight = d.get("mixed3_weight", DeepDream.MIXED3_WEIGHT)
        mixed4_weight = d.get("mixed4_weight", DeepDream.MIXED4_WEIGHT)
        mixed5_weight = d.get("mixed5_weight", DeepDream.MIXED5_WEIGHT)

        return DeepDream(base_image_path, result_prefix,
                         step=step, num_octave=num_octave, octave_scale=octave_scale,
                         iterations=iterations, max_loss=max_loss,
                         mixed2_weight=mixed2_weight, mixed3_weight=mixed3_weight,
                         mixed4_weight=mixed4_weight, mixed5_weight=mixed5_weight)

    def do_dream(self) -> None:  # noqa: C901
        """
        DeepDream algorithm as implemented in
        https://github.com/keras-team/keras/blob/master/examples/deep_dream.py.

        Process:

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

        logger.info(f"Input image: {self.base_image_path}"
                    f" Output image: {self.result_prefix}.png")

        logger.info(f"Initiating deep dream with the following parameters:\n"
                    f"step={self.step}\n"
                    f"num_octave={self.num_octave}\n"
                    f"octave_scale={self.octave_scale}\n"
                    f"iterations={self.iterations}\n"
                    f"max_loss={self.max_loss}\n"
                    f"mixed2_weight={self.mixed2_weight}\n"
                    f"mixed3_weight={self.mixed3_weight}\n"
                    f"mixed4_weight={self.mixed4_weight}\n"
                    f"mixed5_weight={self.mixed5_weight}\n")

        def preprocess_image(image_path: str) -> np.ndarray:
            # Util function to open, resize and format pictures
            # into appropriate tensors.
            image: np.ndarray = load_img(image_path)
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            return inception_v3.preprocess_input(image)

        def deprocess_image(tensor: np.ndarray) -> np.ndarray:
            # Util function to convert a tensor into a valid image.
            if K.image_data_format() == "channels_first":
                tensor = tensor.reshape((3, tensor.shape[2], tensor.shape[3]))
                tensor = tensor.transpose((1, 2, 0))
            else:
                tensor = tensor.reshape((tensor.shape[1], tensor.shape[2], 3))
            tensor /= 2.
            tensor += 0.5
            tensor *= 255.
            tensor = np.clip(tensor, 0, 255).astype("uint8")
            return tensor

        def eval_loss_and_grads(tensor: np.ndarray) -> Tuple[float, np.ndarray]:
            outs: List = fetch_loss_and_grads([tensor])
            loss_value: float = outs[0]
            grad_values: np.ndarray = outs[1]
            return loss_value, grad_values

        def resize_img(image: np.ndarray, size: Tuple) -> np.ndarray:
            image = np.copy(image)
            if K.image_data_format() == "channels_first":
                factors: Tuple = (1, 1,
                                  float(size[0]) / image.shape[2],
                                  float(size[1]) / image.shape[3])
            else:
                factors = (1,
                           float(size[0]) / image.shape[1],
                           float(size[1]) / image.shape[2],
                           1)
            return scipy.ndimage.zoom(image, factors, order=1)

        def gradient_ascent(tensor: np.ndarray, step: float, max_iterations: int,
                            max_loss: Optional[float] = None) -> np.ndarray:
            mod: float = round(max_iterations / 10, -1)  # set loss logging frequency according to max_iterations
            if mod == 0:
                mod = 1
            for i_ in range(max_iterations):
                loss_value, grad_values = eval_loss_and_grads(tensor)
                if max_loss is not None and loss_value > max_loss:
                    break
                if i_ % mod == 0:
                    logger.info(f"..Loss value at {i_}: {loss_value}")
                tensor += step * grad_values
            return tensor

        layer_loss_weight: Dict[str, Dict[str, float]] = {
            "features": {
                "mixed2": self.mixed2_weight,
                "mixed3": self.mixed3_weight,
                "mixed4": self.mixed4_weight,
                "mixed5": self.mixed5_weight,
            },
        }

        K.set_learning_phase(0)

        model: keras.Model = inception_v3.InceptionV3(weights="imagenet",
                                                      include_top=False)
        dream: tf.Tensor = model.input
        logger.info("Model loaded.")

        layer_dict: Dict[str, keras.layers.Layer] = {layer.name: layer for layer in model.layers}

        # Define the loss.
        loss: tf.Tensor = K.variable(0.)
        for layer_name in layer_loss_weight["features"]:
            # Add the L2 norm of the features of a layer to the loss.
            if layer_name not in layer_dict:
                raise ValueError(f"Layer {layer_name} not found in model.")
            coeff: float = layer_loss_weight["features"][layer_name]
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

        img = preprocess_image(self.base_image_path)
        if K.image_data_format() == "channels_first":
            original_shape: Tuple = img.shape[2:]
        else:
            original_shape = img.shape[1:3]
        successive_shapes: List[Tuple] = [original_shape]
        for i in range(1, self.num_octave):
            shape: Tuple = tuple(int(dim / (self.octave_scale ** i)) for dim in original_shape)
            successive_shapes.append(shape)
        successive_shapes = successive_shapes[::-1]
        original_img: np.ndarray = np.copy(img)
        shrunk_original_img: np.ndarray = resize_img(img, successive_shapes[0])

        for shape in successive_shapes:
            logger.info(f"Processing image shape: {shape}")
            img = resize_img(img, shape)
            img = gradient_ascent(img,
                                  step=self.step,
                                  max_iterations=self.iterations,
                                  max_loss=self.max_loss)
            upscaled_shrunk_original_img: np.ndarray = resize_img(shrunk_original_img, shape)
            same_size_original: np.ndarray = resize_img(original_img, shape)
            lost_detail: np.ndarray = same_size_original - upscaled_shrunk_original_img

            img += lost_detail
            shrunk_original_img = resize_img(original_img, shape)

        save_img(self.result_prefix + ".png", deprocess_image(np.copy(img)))


if __name__ == "__main__":

    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Deep Dreams with Keras.")
    parser.add_argument("base_image_path", metavar="base_image_path", type=str, help="Path to the image to transform.")
    parser.add_argument("result_prefix", metavar="result_prefix", type=str, help="Prefix for the saved results.")

    parser.add_argument("--step", type=float, default=DeepDream.STEP,
                        help="Gradient ascent step size.")
    parser.add_argument("--num_octave", type=int, default=DeepDream.NUM_OCTAVE,
                        help="Number of scales at which to run gradient ascent.")
    parser.add_argument("--octave_scale", type=float, default=DeepDream.OCTAVE_SCALE,
                        help="Size ratio between scales.")
    parser.add_argument("--iterations", type=int, default=DeepDream.ITERATIONS,
                        help="Number of ascent steps per scale.")
    parser.add_argument("--max_loss", type=float, default=DeepDream.MAX_LOSS,
                        help="Max allowed loss.")

    parser.add_argument("--mixed2_weight", type=float, default=DeepDream.MIXED2_WEIGHT,
                        help="Mixed layer 2 loss weight.")
    parser.add_argument("--mixed3_weight", type=float, default=DeepDream.MIXED3_WEIGHT,
                        help="Mixed layer 3 loss weight.")
    parser.add_argument("--mixed4_weight", type=float, default=DeepDream.MIXED4_WEIGHT,
                        help="Mixed layer 4 loss weight.")
    parser.add_argument("--mixed5_weight", type=float, default=DeepDream.MIXED5_WEIGHT,
                        help="Mixed layer 5 loss weight.")

    args: argparse.Namespace = parser.parse_args()
    base_image_path_: str = args.base_image_path
    result_prefix_: str = args.result_prefix

    step_: float = args.step
    num_octave_: int = args.num_octave
    octave_scale_: float = args.octave_scale
    iterations_: int = args.iterations
    max_loss_: float = args.max_loss

    mixed2_weight_: float = args.mixed2_weight
    mixed3_weight_: float = args.mixed3_weight
    mixed4_weight_: float = args.mixed4_weight
    mixed5_weight_: float = args.mixed5_weight

    deepdream: DeepDream = DeepDream(base_image_path_,
                                     result_prefix_,
                                     step=step_,
                                     num_octave=num_octave_,
                                     octave_scale=octave_scale_,
                                     iterations=iterations_,
                                     max_loss=max_loss_,
                                     mixed2_weight=mixed2_weight_,
                                     mixed3_weight=mixed3_weight_,
                                     mixed4_weight=mixed4_weight_,
                                     mixed5_weight=mixed5_weight_)
    deepdream.do_dream()
