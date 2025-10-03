import tensorflow as tf
import h5py
import numpy as np
from fomo_trainer import create_fomo_td_with_rnn_combined_model

INPUT_HEIGHT = 96
INPUT_WIDTH = 96
INPUT_CHANNELS = 3
SEQUENCE_LENGTH = 10
COMBINED_MODEL_INPUT_SHAPE = (SEQUENCE_LENGTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
FOMO_BACKBONE_CUTOFF_LAYER_NAME = 'block_3_expand_relu'

def diagnose_mismatch(h5_file):
    """Diagnose weight shape mismatches between saved weights and model architecture"""
    
    print("=" * 80)
    print("DIAGNOSING WEIGHT SHAPE MISMATCH")
    print("=" * 80)
    
    # Extract saved weight shapes
    print("\n1. SAVED WEIGHTS IN H5 FILE:")
    print("-" * 80)
    saved_weights = {}
    with h5py.File(h5_file, 'r') as f:
        def extract_weights(name, obj):
            if isinstance(obj, h5py.Dataset) and 'model_weights' in name:
                # Clean up the path to get layer name and weight name
                parts = name.replace('model_weights/', '').split('/')
                if len(parts) >= 2:
                    layer_name = parts[0]
                    weight_name = '/'.join(parts[1:])
                    if layer_name not in saved_weights:
                        saved_weights[layer_name] = {}
                    saved_weights[layer_name][weight_name] = obj.shape
                    print(f"  {layer_name:40s} | {weight_name:30s} | shape: {obj.shape}")
        
        f.visititems(extract_weights)
    
    # Analyze ConvLSTM weights specifically
    print("\n2. CONVLSTM WEIGHT ANALYSIS:")
    print("-" * 80)
    if 'motion_convlstm' in saved_weights:
        lstm_weights = saved_weights['motion_convlstm']
        for weight_name, shape in lstm_weights.items():
            print(f"  {weight_name}: {shape}")
            if 'kernel:0' in weight_name and 'recurrent' not in weight_name:
                print(f"    → Input kernel: filter_size=({shape[0]}, {shape[1]}), input_channels={shape[2]}, output_filters={shape[3]}")
            elif 'recurrent_kernel:0' in weight_name:
                print(f"    → Recurrent kernel: filter_size=({shape[0]}, {shape[1]}), state_channels={shape[2]}, output_filters={shape[3]}")
            elif 'bias:0' in weight_name:
                print(f"    → Bias: {shape[0]} units (= output_filters)")
    
    # Try different filter configurations
    print("\n3. TESTING DIFFERENT MODEL CONFIGURATIONS:")
    print("-" * 80)
    
    for filters in [4, 8, 16]:
        print(f"\nTrying rnn_conv_lstm_filters={filters}:")
        try:
            model_params = {
                "input_sequence_shape": COMBINED_MODEL_INPUT_SHAPE,
                "fomo_num_classes": 1,
                "motion_num_classes": 1,
                "fomo_alpha": 0.35,
                "fomo_backbone_cutoff_layer_name": FOMO_BACKBONE_CUTOFF_LAYER_NAME,
                "rnn_type": "convlstm",
                "rnn_conv_lstm_filters": filters,
                "rnn_dense_units": 8,
                "rnn_gru_units": 32,
            }
            
            model = create_fomo_td_with_rnn_combined_model(**model_params)
            
            # Find the ConvLSTM layer
            for layer in model.layers:
                if 'convlstm' in layer.name.lower():
                    print(f"  ✓ Model ConvLSTM output shape: {layer.output_shape}")
                    print(f"    Weights:")
                    for weight in layer.weights:
                        print(f"      - {weight.name}: {weight.shape}")
                    
                    # Try to match with saved weights
                    matches = True
                    if 'motion_convlstm' in saved_weights:
                        for saved_name, saved_shape in saved_weights['motion_convlstm'].items():
                            weight_match = None
                            for w in layer.weights:
                                if saved_name.split('/')[-1] in w.name:
                                    weight_match = w
                                    break
                            
                            if weight_match:
                                model_shape = tuple(weight_match.shape.as_list())
                                if model_shape != saved_shape:
                                    print(f"      ✗ MISMATCH: {saved_name}")
                                    print(f"        Saved:  {saved_shape}")
                                    print(f"        Model:  {model_shape}")
                                    matches = False
                            else:
                                print(f"      ? No matching weight found for {saved_name}")
                                matches = False
                    
                    if matches:
                        print(f"  ✓✓ ALL WEIGHTS MATCH! Use rnn_conv_lstm_filters={filters}")
                    
                    break
                    
        except Exception as e:
            print(f"  ✗ Error creating model: {e}")
    
    # Check dense layer too
    print("\n4. DENSE LAYER ANALYSIS:")
    print("-" * 80)
    if 'motion_dense_layer' in saved_weights:
        dense_weights = saved_weights['motion_dense_layer']
        for weight_name, shape in dense_weights.items():
            if 'kernel:0' in weight_name:
                input_size = shape[0]
                output_size = shape[1]
                print(f"  Dense layer expects input_size={input_size}, output_size={output_size}")
                
                # Calculate what ConvLSTM output would give this input size
                # input_size = height * width * filters
                # We know height=24, width=24 (from model summary)
                filters_needed = input_size / (24 * 24)
                print(f"  → This means ConvLSTM output should have {filters_needed} filters")
                print(f"  → (24 × 24 × {filters_needed} = {input_size})")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python diagnose_weight_mismatch.py <h5_file>")
        sys.exit(1)
    
    h5_file = sys.argv[1]
    diagnose_mismatch(h5_file)

