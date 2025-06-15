import os
import glob
import numpy as np
import tensorflow as tf
import json # Importer le module json
from fomo_trainer import create_fomo_model, fomo_loss_function, FomoDataGenerator

# --- Configuration ---
DATA_BASE_DIR = 'data/sport_MOT_sample'
PERSON_CLASS_ID = 0  # Assuming 'person' is class 0 in annotation files
NUM_OBJECT_CLASSES = 1  # We are only detecting one class: 'person'

# Model and Training Hyperparameters (adjust as needed)
INPUT_HEIGHT = 96
INPUT_WIDTH = 96
INPUT_CHANNELS = 3 # MobileNetV2 expects 3 channels
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
ALPHA = 0.35  # MobileNetV2 alpha

# Backbone cutoff layer and spatial reduction
# Default: 'block_6_expand_relu' for 1/8th reduction (96x96 -> 12x12 grid)
# Alternative 1: 'block_3_expand_relu' for 1/4th reduction (96x96 -> 24x24 grid)
# Alternative 2: 'expanded_conv_project_BN' for 1/2 reduction (96x96 -> 48x48 grid)
# Alternative 3: 'block_10_expand_relu' for 1/16 reduction (96x96 -> 6x6 grid)
BACKBONE_CUTOFF_LAYER_NAME = 'block_6_expand_relu' # MODIFIEZ CECI pour une autre option
BACKBONE_CUTOFF_LAYER_NAME = 'block_10_expand_relu'

_known_cutoffs = {
    'expanded_conv_project_BN': 2,   # Results in 1/2 spatial reduction (e.g., 96x96 -> 48x48)
    'block_3_expand_relu': 4,       # Results in 1/4 spatial reduction (e.g., 96x96 -> 24x24)
    'block_6_expand_relu': 8,       # Results in 1/8 spatial reduction (e.g., 96x96 -> 12x12)
    'block_10_expand_relu': 16      # Results in 1/16 spatial reduction (e.g., 96x96 -> 6x6)
}

if BACKBONE_CUTOFF_LAYER_NAME not in _known_cutoffs:
    raise ValueError(
        f"Unsupported BACKBONE_CUTOFF_LAYER_NAME: {BACKBONE_CUTOFF_LAYER_NAME}. "
        f"Supported are: {list(_known_cutoffs.keys())}"
    )
SPATIAL_REDUCTION = _known_cutoffs[BACKBONE_CUTOFF_LAYER_NAME]

GRID_HEIGHT = INPUT_HEIGHT // SPATIAL_REDUCTION
GRID_WIDTH = INPUT_WIDTH // SPATIAL_REDUCTION
GRID_SHAPE = (GRID_HEIGHT, GRID_WIDTH)

BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 50  # Start with a smaller number for testing, e.g., 10-50
OBJECT_WEIGHT = 100.0
MODEL_SAVE_PATH = 'person_detector_fomo.h5' # Keras H5 format
# --- End Configuration ---

def load_annotations(ann_file_path):
    """
    Loads annotations from a JSON file.
    Extracts centroids for objects with classTitle 'person'.
    """
    annotations = []
    if not os.path.exists(ann_file_path):
        return annotations

    try:
        with open(ann_file_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: Could not decode JSON from {ann_file_path}")
        return annotations
    except Exception as e:
        print(f"Warning: Error reading {ann_file_path}: {e}")
        return annotations

    image_width = data.get('size', {}).get('width')
    image_height = data.get('size', {}).get('height')

    if not image_width or not image_height:
        print(f"Warning: Image dimensions not found in {ann_file_path}")
        return annotations

    for obj in data.get('objects', []):
        if obj.get('classTitle') == 'person': # Assuming PERSON_CLASS_ID corresponds to 'person'
            points = obj.get('points', {}).get('exterior')
            if points and len(points) == 2 and len(points[0]) == 2 and len(points[1]) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                
                center_x_pixel = (x1 + x2) / 2
                center_y_pixel = (y1 + y2) / 2
                
                x_center_norm = center_x_pixel / image_width
                y_center_norm = center_y_pixel / image_height
                
                # Ensure normalized coordinates are within [0, 1]
                x_center_norm = min(max(x_center_norm, 0.0), 1.0)
                y_center_norm = min(max(y_center_norm, 0.0), 1.0)

                # FomoDataGenerator expects class_id to be 0 for the first (and in this case, only) class
                annotations.append((x_center_norm, y_center_norm, PERSON_CLASS_ID))
                print(f"Loaded person annotation: ({x_center_norm}, {y_center_norm}) from {ann_file_path}")
            else:
                print(f"Warning: Malformed points for a person object in {ann_file_path}")
    return annotations

def load_dataset_split(base_dir, split_name):
    """
    Loads image paths and annotations for a given data split (train, val, test).
    """
    img_dir = os.path.join(base_dir, split_name, 'img')
    ann_dir = os.path.join(base_dir, split_name, 'ann')

    image_paths = []
    all_annotations = []

    # Consider common image extensions
    for ext in ('*.jpg', '*.jpeg', '*.png'):
        image_paths.extend(glob.glob(os.path.join(img_dir, ext)))
    
    image_paths = sorted(image_paths) # Ensure consistent order

    for img_path in image_paths:
        img_basename = os.path.basename(img_path)
        # Annotation filename is image_filename.json (e.g., v_1LwtoLPw2TU_c006_000234.jpg.json)
        ann_filename = img_basename + '.json'
        ann_path = os.path.join(ann_dir, ann_filename)
        
        annotations_for_image = load_annotations(ann_path)
        # Even if annotations are empty (no person in image), we keep the image for background learning
        all_annotations.append(annotations_for_image)
            
    print(f"Loaded {len(image_paths)} images and their annotations for split '{split_name}'.")
    return image_paths, all_annotations


if __name__ == '__main__':
    print("Loading training data...")
    train_image_files, train_annotations = load_dataset_split(DATA_BASE_DIR, 'train')
    
    print("Loading validation data...")
    val_image_files, val_annotations = load_dataset_split(DATA_BASE_DIR, 'val')

    if not train_image_files:
        print("Error: No training images found. Please check DATA_BASE_DIR and 'train' subdirectory.")
        exit()
    if not val_image_files:
        print("Warning: No validation images found. Training will proceed without validation.")

    # Create data generators
    train_generator = FomoDataGenerator(
        image_filenames=train_image_files,
        annotations=train_annotations,
        batch_size=BATCH_SIZE,
        input_shape=INPUT_SHAPE,
        grid_shape=GRID_SHAPE,
        num_classes=NUM_OBJECT_CLASSES, # Number of object classes (e.g., 1 for person)
        shuffle=True
    )

    validation_generator = None
    if val_image_files:
        validation_generator = FomoDataGenerator(
            image_filenames=val_image_files,
            annotations=val_annotations,
            batch_size=BATCH_SIZE,
            input_shape=INPUT_SHAPE,
            grid_shape=GRID_SHAPE,
            num_classes=NUM_OBJECT_CLASSES,
            shuffle=False # No need to shuffle validation data
        )

    # Create the FOMO model
    # NUM_OBJECT_CLASSES is the number of distinct object types (e.g., 1 for 'person')
    # The background class is handled internally by the model and loss function.
    model = create_fomo_model(
        INPUT_SHAPE, 
        NUM_OBJECT_CLASSES, 
        alpha=ALPHA,
        backbone_cutoff_layer_name=BACKBONE_CUTOFF_LAYER_NAME
    )
    model.summary()

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # num_classes_with_background for loss is NUM_OBJECT_CLASSES + 1
    loss_fn = fomo_loss_function(
        object_weight=OBJECT_WEIGHT, 
        num_classes_with_background=NUM_OBJECT_CLASSES + 1 
    )
    
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy']) # Added accuracy for basic monitoring

    print(f"\nStarting training for person detection...")
    print(f"Input shape: {INPUT_SHAPE}, Grid shape: {GRID_SHAPE}")
    print(f"Number of object classes (excluding background): {NUM_OBJECT_CLASSES}")
    print(f"Training on {len(train_image_files)} images, validating on {len(val_image_files) if val_image_files else 0} images.")

    try:
        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss' if validation_generator else 'loss'),
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss' if validation_generator else 'loss', factor=0.2, patience=5, min_lr=0.00001),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss' if validation_generator else 'loss', patience=10, restore_best_weights=True)
            ]
        )
        print("\nTraining completed.")
        print(f"Model saved to {MODEL_SAVE_PATH} (if validation improved or no validation).")

        # You might want to plot training history here: history.history['loss'], history.history['val_loss'] etc.

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")

    # To load the model later:
    # loaded_model = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects={'loss': loss_fn})
    # print("Model loaded successfully.")
