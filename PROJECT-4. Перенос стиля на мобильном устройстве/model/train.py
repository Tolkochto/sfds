import tensorflow as tf
import matplotlib.pyplot as plt
import os

IMG_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 300


# -----------------------------
# Image Loading
# -----------------------------
def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img


def get_image_paths(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(("jpg", "jpeg", "png"))
    ]


content_paths = get_image_paths("datasets/content")
style_paths = get_image_paths("datasets/style")


# -----------------------------
# Dataset
# -----------------------------
def create_dataset(content_paths):
    ds = tf.data.Dataset.from_tensor_slices(content_paths)
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(100).batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
    return ds


dataset = create_dataset(content_paths)


# -----------------------------
# Custom Layers
# -----------------------------
class InstanceNormalization(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],), initializer="ones", trainable=True
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],), initializer="zeros", trainable=True
        )

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x = (x - mean) / tf.sqrt(var + 1e-5)
        return self.gamma * x + self.beta


class ReflectionPad2D(tf.keras.layers.Layer):
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def call(self, x):
        p = self.padding
        return tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], mode="REFLECT")


# -----------------------------
# Generator
# -----------------------------
def build_generator():
    inputs = tf.keras.layers.Input(shape=(256, 256, 3))

    x = ReflectionPad2D(4)(inputs)
    x = tf.keras.layers.Conv2D(32, 9, padding="valid")(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Downsample
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Residual blocks
    for _ in range(5):
        res = x
        x = ReflectionPad2D(1)(x)
        x = tf.keras.layers.Conv2D(128, 3, padding="valid")(x)
        x = InstanceNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)

        x = ReflectionPad2D(1)(x)
        x = tf.keras.layers.Conv2D(128, 3, padding="valid")(x)
        x = InstanceNormalization()(x)

        x = tf.keras.layers.Add()([x, res])

    # Upsample
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = InstanceNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = ReflectionPad2D(4)(x)
    x = tf.keras.layers.Conv2D(3, 9, padding="valid", activation="sigmoid")(x)

    return tf.keras.Model(inputs, x)


generator = build_generator()


# -----------------------------
# VGG Feature Extractor
# -----------------------------
content_layer = "block4_conv2"
style_layers = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]


def build_vgg_extractor():
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in style_layers + [content_layer]]
    return tf.keras.Model(vgg.input, outputs)


vgg_extractor = build_vgg_extractor()


# -----------------------------
# Style Image
# -----------------------------
style_img = load_image(style_paths[0])
style_img = tf.expand_dims(style_img, 0)

style_vgg = tf.keras.applications.vgg19.preprocess_input(style_img * 255)
style_targets = vgg_extractor(style_vgg)


# -----------------------------
# Gram Matrix
# -----------------------------
def gram_matrix(x):
    shape = tf.shape(x)
    features = tf.reshape(x, [shape[0], -1, shape[3]])
    gram = tf.matmul(features, features, transpose_a=True)
    return gram / tf.cast(shape[1] * shape[2], tf.float32)


# -----------------------------
# Loss
# -----------------------------
def compute_style_loss(gen_features):

    loss = 0

    for g, s in zip(gen_features[:-1], style_targets[:-1]):
        gram_g = gram_matrix(g)
        gram_s = gram_matrix(s)

        loss += tf.reduce_mean((gram_g - gram_s) ** 2)

    return loss / len(style_layers)


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)


# -----------------------------
# Train Step
# -----------------------------
@tf.function
def train_step(content):

    with tf.GradientTape() as tape:
        generated = generator(content)

        gen_vgg = tf.keras.applications.vgg19.preprocess_input(generated * 255)
        content_vgg = tf.keras.applications.vgg19.preprocess_input(content * 255)

        gen_outputs = vgg_extractor(gen_vgg)
        content_outputs = vgg_extractor(content_vgg)

        content_loss = tf.reduce_mean((gen_outputs[-1] - content_outputs[-1]) ** 2)
        style_loss = compute_style_loss(gen_outputs)

        tv_loss = tf.reduce_mean(tf.image.total_variation(generated))

        total_loss = 1 * content_loss + 200 * style_loss + 1e-3 * tv_loss

    grads = tape.gradient(total_loss, generator.trainable_variables)
    optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    return total_loss


# -----------------------------
# Visualization
# -----------------------------
os.makedirs("visualizations", exist_ok=True)


def visualize(content, generated, epoch):

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(content[0])
    plt.title("Content")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(style_img[0])
    plt.title("Style")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(generated[0])
    plt.title("Generated")
    plt.axis("off")

    plt.savefig(f"visualizations/epoch_{epoch}.png")
    plt.close()


# -----------------------------
# Training Loop
# -----------------------------
steps_per_epoch = len(content_paths) // BATCH_SIZE

for epoch in range(EPOCHS):
    for step, content_img in enumerate(dataset.take(steps_per_epoch)):
        loss = train_step(content_img)

    generated = generator(content_img)

    visualize(content_img, generated, epoch)

    print("Epoch:", epoch, "Loss:", loss.numpy())
