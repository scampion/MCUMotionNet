import tensorflow as tf

# This will download and cache the weights
model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),  # or your input shape
    alpha=0.35,  # or your alpha value
    include_top=False,
    weights='imagenet'
)

# Save the weights
model.save_weights('mobilenet_v2.weights.h5')


