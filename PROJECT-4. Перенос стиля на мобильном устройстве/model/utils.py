import tensorflow as tf
import os


def load_image(path, size=(256, 256)):
    """Loads and resizes an image"""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, size)
    img = img[tf.newaxis, ...]  # Add batch dimension
    return img


def create_dataset(content_dir, style_dir):
    """Creates datasets by loading content and style images"""
    content_images = [os.path.join(content_dir, f) for f in os.listdir(content_dir)]
    style_images = [os.path.join(style_dir, f) for f in os.listdir(style_dir)]
    return content_images, style_images
