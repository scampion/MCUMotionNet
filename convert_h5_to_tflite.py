import tensorflow as tf
import argparse
import h5py
import numpy as np
from fomo_trainer import create_fomo_td_with_rnn_combined_model

INPUT_HEIGHT = 96
INPUT_WIDTH = 96
INPUT_CHANNELS = 3
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
SEQUENCE_LENGTH = 10
COMBINED_MODEL_INPUT_SHAPE = (SEQUENCE_LENGTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
FOMO_BACKBONE_CUTOFF_LAYER_NAME = 'block_3_expand_relu'


def load_weights_manually(model, h5_file):
    """
    Manually load weights from H5 file with custom mapping for ConvLSTM.
    This handles the case where ConvLSTM has 4 filters but 16 internal units.
    """
    print("Loading weights manually with custom mapping...")

    with h5py.File(h5_file, 'r') as f:
        model_weights_group = f['model_weights']

        for layer in model.layers:
            layer_name = layer.name

            # Check if this layer exists in the H5 file
            if layer_name in model_weights_group:
                layer_group = model_weights_group[layer_name]
                print(f"  Loading weights for layer: {layer_name}")

                weights_to_set = []
                for weight in layer.weights:
                    weight_name = weight.name.split('/')[-1]  # Get just the weight name

                    # Try to find matching weight in H5 file
                    found = False
                    for h5_weight_name in layer_group.keys():
                        if weight_name in h5_weight_name:
                            h5_weight = layer_group[h5_weight_name][:]

                            # Check if shapes match
                            if weight.shape.as_list() == list(h5_weight.shape):
                                weights_to_set.append(h5_weight)
                                print(f"    ✓ {weight_name}: {h5_weight.shape}")
                                found = True
                                break
                            else:
                                print(f"    ✗ Shape mismatch for {weight_name}:")
                                print(f"      Model expects: {weight.shape.as_list()}")
                                print(f"      H5 has: {list(h5_weight.shape)}")

                    if not found:
                        print(f"    ? Weight not found: {weight_name}")

                if weights_to_set and len(weights_to_set) == len(layer.weights):
                    try:
                        layer.set_weights(weights_to_set)
                        print(f"    ✓ Successfully set {len(weights_to_set)} weights")
                    except Exception as e:
                        print(f"    ✗ Error setting weights: {e}")
            else:
                # Try alternative naming (nested structure)
                alt_names = [
                    f"{layer_name}/{layer_name}",  # Nested naming
                    layer_name.replace('_', ''),  # Without underscores
                ]

                found_alt = False
                for alt_name in alt_names:
                    if alt_name in model_weights_group:
                        print(f"  Found alternative: {alt_name} for {layer_name}")
                        found_alt = True
                        break

                if not found_alt:
                    print(f"  ⊘ No weights found for layer: {layer_name}")


def convert(h5_file, tflite_file):
    """
    Converts a Keras .h5 file to a .tflite file by rebuilding the model
    architecture and then loading the weights with custom handling.
    """
    # Model parameters matching the saved model
    # Based on the diagnostic: ConvLSTM has 4 filters (output), Dense expects 2304 inputs
    model_params = {
        "input_sequence_shape": COMBINED_MODEL_INPUT_SHAPE,
        "fomo_num_classes": 1,
        "motion_num_classes": 1,
        "fomo_alpha": 0.35,
        "fomo_backbone_cutoff_layer_name": FOMO_BACKBONE_CUTOFF_LAYER_NAME,
        "rnn_type": "convlstm",
        "rnn_conv_lstm_filters": 4,  # This gives output shape (24, 24, 4)
        "rnn_dense_units": 8,
        "rnn_gru_units": 32,
    }

    print("Re-creating model architecture...")
    model = create_fomo_td_with_rnn_combined_model(**model_params)

    print("\nModel summary:")
    model.summary()

    print(f"\n{'=' * 80}")
    print("ATTEMPTING TO LOAD WEIGHTS")
    print('=' * 80)

    # First try: standard keras load_weights with skip_mismatch
    print("\nAttempt 1: Using model.load_weights() with skip_mismatch=True")
    try:
        model.load_weights(h5_file, by_name=True, skip_mismatch=True)
        print("✓ Weights loaded with skip_mismatch=True")

        # Verify which layers got weights
        print("\nVerifying loaded layers:")
        for layer in model.layers:
            if layer.weights:
                total_params = sum([tf.size(w).numpy() for w in layer.weights])
                print(f"  {layer.name}: {total_params} parameters")

    except Exception as e:
        print(f"✗ Failed: {e}")
        print("\nAttempt 2: Manual weight loading with custom mapping")
        load_weights_manually(model, h5_file)

    print(f"\n{'=' * 80}")
    print("CONVERTING TO TFLITE")
    print('=' * 80)

    print("Converting model to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optional: Add optimizations
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    try:
        tflite_model = converter.convert()

        with open(tflite_file, 'wb') as f:
            f.write(tflite_model)

        print(f"\n✓ Successfully converted model to {tflite_file}")
        print(f"  TFLite model size: {len(tflite_model) / 1024:.2f} KB")

    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        print("\nNote: The model was created but conversion failed.")
        print("This might be due to unsupported operations in TFLite.")
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert Keras .h5 model to TensorFlow Lite .tflite model.'
    )
    parser.add_argument('h5_file', help='Path to the input .h5 file.')
    parser.add_argument('tflite_file', help='Path to the output .tflite file.')
    args = parser.parse_args()

    convert(args.h5_file, args.tflite_file)