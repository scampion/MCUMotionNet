
import tensorflow as tf
import argparse
from fomo_trainer import create_fomo_td_with_rnn_combined_model

def convert(h5_file, tflite_file):
    """
    Converts a Keras .h5 file to a .tflite file by rebuilding the model
    architecture and then loading the weights. This is a robust method
    to avoid deserialization errors with custom layers like Lambda.
    """
    # These parameters must match the configuration of the model that was saved.
    # They are taken from the 'small_rnn_lstm_filters_4_dense_8' experiment
    # in train_combined_model_stage2.py.
    model_params = {
        "input_sequence_shape": (10, 96, 96, 3),
        "fomo_num_classes": 1,
        "motion_num_classes": 1,
        "fomo_alpha": 0.35,
        "fomo_backbone_cutoff_layer_name": 'block_3_expand_relu',
        "rnn_type": "convlstm",
        "rnn_conv_lstm_filters": 4,
        "rnn_dense_units": 8,
        "rnn_gru_units": 32,  # Not used for convlstm but required by function signature
    }

    print("Re-creating model architecture...")
    model = create_fomo_td_with_rnn_combined_model(**model_params)

    print(f"Loading weights from {h5_file}...")
    # Add by_name=True to match layers by name instead of by order.
    # This is a robust way to load weights when the model architecture
    # is complex and small differences can lead to topology mismatches.
    model.load_weights(h5_file, by_name=True)

    print("Converting model to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(tflite_file, 'wb') as f:
        f.write(tflite_model)
    print(f"Successfully converted model to {tflite_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Keras .h5 model to TensorFlow Lite .tflite model by rebuilding the architecture.'
    )
    parser.add_argument('h5_file', help='Path to the input .h5 file.')
    parser.add_argument('tflite_file', help='Path to the output .tflite file.')
    args = parser.parse_args()

    convert(args.h5_file, args.tflite_file)
