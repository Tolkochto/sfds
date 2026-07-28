import tensorflow as tf
import tensorflow_hub as hub
import os
from utils import load_image, create_dataset

# ================= LOAD MODEL =================
hub_model_url = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
style_model = hub.load(hub_model_url)

# ================= DATA =================
content_dir = "datasets/content"
style_dir = "datasets/style"
content_images, style_images = create_dataset(content_dir, style_dir)

# ================= TRAINING SETUP =================
learning_rate = 1e-3
optimizer = tf.keras.optimizers.Adam(learning_rate)
style_weight = tf.Variable(1.0)

# ================= PATHS =================
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("exported_model", exist_ok=True)


# ================= EXPORT MODULE =================
class StyleTransferModule(tf.Module):
    def __init__(self, hub_model, style_weight):
        super().__init__()
        self.hub_model = hub_model
        self.style_weight = style_weight

    @tf.function(
        input_signature=[
            tf.TensorSpec([1, 256, 256, 3], tf.float32),
            tf.TensorSpec([1, 256, 256, 3], tf.float32),
        ]
    )
    def __call__(self, content, style):
        stylized = self.hub_model(content, style)[0]
        output = content + self.style_weight * (stylized - content)
        return {"output": output}


# ================= TRAIN LOOP =================
for content_idx, c_path in enumerate(content_images):
    content = load_image(c_path)

    for style_idx, s_path in enumerate(style_images):
        style = load_image(s_path)

        with tf.GradientTape() as tape:
            stylized = style_model(content, style)[0]
            loss = tf.reduce_mean((stylized - content) ** 2) * style_weight

        grads = tape.gradient(loss, [style_weight])
        optimizer.apply_gradients(zip(grads, [style_weight]))

    checkpoint = tf.train.Checkpoint(style_weight=style_weight, optimizer=optimizer)
    checkpoint.save(os.path.join("checkpoints", "ckpt"))

# ================= EXPORT MODEL =================
print("\nExporting SavedModel...")

export_module = StyleTransferModule(style_model, style_weight)

tf.saved_model.save(
    export_module,
    "exported_model",
    signatures={"serving_default": export_module.__call__},
)

print("✅ SavedModel exported")

# ================= CONVERT TO TFLITE =================
print("Converting to TFLite...")

converter = tf.lite.TFLiteConverter.from_saved_model("exported_model")

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]

tflite_model = converter.convert()

with open("style_transfer.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ TFLite model saved: style_transfer.tflite")
print("\nTraining complete!")
