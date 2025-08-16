import os
import glob
import cv2
import numpy as np
import tensorflow as tf
import math
import json
import pandas as pd # Ajout de pandas pour lire les CSV
from tensorflow.keras.utils import Sequence as KerasSequence
from joblib import Memory
import datetime

memory = Memory("./cachedir", verbose=0)

from fomo_trainer import (
    create_fomo_td_with_rnn_combined_model,
    fomo_loss_function
)

# --- Experiment Configurations ---
EXPERIMENTS = [
    {
        "name": "extra_small_rnn_lstm_filters_2_dense_4",
        "rnn_type": "convlstm",
        "rnn_conv_lstm_filters": 2,
        "rnn_dense_units": 4,
        "rnn_gru_units": 16,  # Not used for convlstm
    },
    {
        "name": "small_rnn_lstm_filters_4_dense_8",
        "rnn_type": "convlstm",
        "rnn_conv_lstm_filters": 4,
        "rnn_dense_units": 8,
        "rnn_gru_units": 32,  # Not used for convlstm
    },
    {
        "name": "medium_rnn_lstm_filters_8_dense_16",
        "rnn_type": "convlstm",
        "rnn_conv_lstm_filters": 8,
        "rnn_dense_units": 16,
        "rnn_gru_units": 64,  # Not used for convlstm
    },
    {
        "name": "large_rnn_lstm_filters_16_dense_16",
        "rnn_type": "convlstm",
        "rnn_conv_lstm_filters": 16,
        "rnn_dense_units": 16,
        "rnn_gru_units": 64,  # Not used for convlstm
    },
    {
        "name": "extra_large_rnn_lstm_filters_32_dense_32",
        "rnn_type": "convlstm",
        "rnn_conv_lstm_filters": 32,
        "rnn_dense_units": 32,
        "rnn_gru_units": 128,  # Not used for convlstm
    }
]


# --- General Configuration ---
VIDEO_DATA_DIR = '/Users/scampion/src/sport_video_scrapper/data/videos'
ANNOTATION_DATA_DIR = '/Users/scampion/src/sport_video_scrapper/camera_movement_reports'

VIDEO_DATA_DIR = os.environ.get("VIDEO_DATA_DIR", VIDEO_DATA_DIR)
ANNOTATION_DATA_DIR = os.environ.get("ANNOTATION_DATA_DIR", ANNOTATION_DATA_DIR)


STAGE1_FOMO_MODEL_PATH = 'person_detector_fomo.h5'
BASE_MODEL_SAVE_NAME = 'fomo_td_rnn_regression_stage2'

COMBINED_MODEL_SAVE_DIR = os.environ.get('COMBINED_MODEL_SAVE_DIR', '.')

if 'COMBINED_MODEL_SAVE_DIR' in os.environ:
    COMBINED_MODEL_SAVE_DIR = os.environ['COMBINED_MODEL_SAVE_DIR']
    if not os.path.exists(COMBINED_MODEL_SAVE_DIR):
        os.makedirs(COMBINED_MODEL_SAVE_DIR)
    COMBINED_MODEL_SAVE_PATH = os.path.join(COMBINED_MODEL_SAVE_DIR, COMBINED_MODEL_SAVE_PATH)



INPUT_HEIGHT = 96
INPUT_WIDTH = 96
INPUT_CHANNELS = 3
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
SEQUENCE_LENGTH = 10
COMBINED_MODEL_INPUT_SHAPE = (SEQUENCE_LENGTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)

FOMO_ALPHA = 0.35
_known_cutoffs_local = {
    'expanded_conv_project_BN': 2,
    'block_3_expand_relu': 4,
    'block_6_expand_relu': 8,
    'block_10_expand_relu': 16
}
FOMO_BACKBONE_CUTOFF_LAYER_NAME = 'block_3_expand_relu'
if FOMO_BACKBONE_CUTOFF_LAYER_NAME not in _known_cutoffs_local:
    raise ValueError(f"Cutoff layer {FOMO_BACKBONE_CUTOFF_LAYER_NAME} non supporté ici.")
SPATIAL_REDUCTION = _known_cutoffs_local[FOMO_BACKBONE_CUTOFF_LAYER_NAME]

GRID_HEIGHT = INPUT_HEIGHT // SPATIAL_REDUCTION
GRID_WIDTH = INPUT_WIDTH // SPATIAL_REDUCTION
NUM_CLASSES_MODEL_OUTPUT_FOMO = 1 + 1

NUM_MOTION_OUTPUTS = 1
ANNOTATION_MOVE_X_NORMALIZATION_FACTOR = 3.0

BATCH_SIZE_STAGE2 = 8
LEARNING_RATE_STAGE2 = 0.0005
EPOCHS_STAGE2 = 30

RNN_TYPE_STAGE2 = 'convlstm' 

# Configuration pour TensorBoard
LOG_DIR = "logs/fit2/" 
if 'TENSORBOARD_LOG_DIR' in os.environ:
    LOG_DIR = os.environ['TENSORBOARD_LOG_DIR']

# --- End of Configuration ---


def preprocess_single_frame(frame, target_shape):
    img_resized = cv2.resize(frame, (target_shape[1], target_shape[0]))
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized

class VideoSequenceDataGenerator(KerasSequence):
    def __init__(self, video_files, annotation_dir, batch_size, sequence_length,
                 input_shape, shuffle=True):
        self.video_files = video_files
        self.annotation_dir = annotation_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.sequences_data = self._extract_sequences_from_videos()
        self.indexes = np.arange(len(self.sequences_data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _extract_sequences_from_videos(self):
        all_sequences_with_labels = []
        print("Extracting sequences and motion labels from CSVs...")
        for video_path in self.video_files:
            video_filename = os.path.basename(video_path)
            video_name_without_ext = os.path.splitext(video_filename)[0]
            annotation_path = os.path.join(self.annotation_dir, f"{video_name_without_ext}_movement.csv")

            if not os.path.exists(annotation_path):
                print(f"Warning: CSV annotation file not found for {video_path} at {annotation_path}. Video skipped.")
                continue

            try:
                annotations_df = pd.read_csv(annotation_path)
                if 'Move_X' not in annotations_df.columns:
                    print(f"Warning: 'Move_X' column not found in {annotation_path}. Video skipped.")
                    continue
                move_x_annotations = annotations_df['Move_X'].tolist()
            except Exception as e:
                print(f"Error reading CSV file {annotation_path}: {e}. Video skipped.")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}. Video skipped.")
                continue

            frame_count_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if frame_count_video < self.sequence_length:
                print(f"Warning: Video {video_path} is too short for sequence length {self.sequence_length}. Video skipped.")
                continue
            
            annotations_len = len(move_x_annotations)
            if frame_count_video > annotations_len:
                 print(f"Warning: Fewer annotations ({annotations_len}) than frames ({frame_count_video}) for {video_path}. Truncating to annotations length.")
                 frame_count_video = annotations_len

            num_possible_sequences = frame_count_video - self.sequence_length + 1
            for i in range(num_possible_sequences):
                label_index = i + self.sequence_length - 1
                motion_label_value = move_x_annotations[label_index]
                if pd.isna(motion_label_value):
                    continue

                clipped_value = np.clip(float(motion_label_value), -ANNOTATION_MOVE_X_NORMALIZATION_FACTOR, ANNOTATION_MOVE_X_NORMALIZATION_FACTOR)
                normalized_motion_value = clipped_value / ANNOTATION_MOVE_X_NORMALIZATION_FACTOR
                motion_label = np.array([normalized_motion_value], dtype=np.float32)
                all_sequences_with_labels.append((video_path, i, motion_label))

        print(f"Extraction complete. {len(all_sequences_with_labels)} sequences generated.")

        if not all_sequences_with_labels:
            print("WARNING: No sequences could be generated from the provided videos and annotations.")
        return all_sequences_with_labels

    def __len__(self):
        return math.ceil(len(self.sequences_data) / self.batch_size)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_sequences = []
        batch_motion_labels = []

        for i in batch_indexes:
            video_path, start_frame_idx, motion_label = self.sequences_data[i]
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
            current_sequence_frames = []
            for _ in range(self.sequence_length):
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = preprocess_single_frame(frame, self.input_shape)
                current_sequence_frames.append(processed_frame)
            cap.release()

            if len(current_sequence_frames) == self.sequence_length:
                batch_sequences.append(np.array(current_sequence_frames))
                batch_motion_labels.append(motion_label)

        if not batch_sequences:
            empty_batch_x_shape = (0, self.sequence_length, self.input_shape[0], self.input_shape[1], self.input_shape[2])
            empty_batch_y_shape = (0, NUM_MOTION_OUTPUTS)
            return np.empty(empty_batch_x_shape), {'motion_output': np.empty(empty_batch_y_shape)}

        return np.array(batch_sequences), {'motion_output': np.array(batch_motion_labels)}

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def run_experiment(config, fomo_stage1_model, train_gen, val_gen):
    """
    Runs a single training experiment with a given configuration, with checkpointing.
    """
    print(f"\n{'='*20} Running Experiment: {config['name']} {'='*20}")
    print(f"Hyperparameters: {config}")

    # Define paths for this experiment's artifacts
    experiment_dir = os.path.join(COMBINED_MODEL_SAVE_DIR, config['name'])
    os.makedirs(experiment_dir, exist_ok=True)
    
    model_checkpoint_path = os.path.join(experiment_dir, f"{BASE_MODEL_SAVE_NAME}.h5")
    epoch_file_path = os.path.join(experiment_dir, "last_epoch.txt")
    tensorboard_log_dir = os.path.join(LOG_DIR, config['name']) # Unified log dir for the experiment

    print(f"Experiment artifacts will be saved in: {experiment_dir}")
    print(f"TensorBoard logs will be saved to: {tensorboard_log_dir}")

    initial_epoch = 0
    if os.path.exists(model_checkpoint_path):
        print(f"Resuming training from checkpoint: {model_checkpoint_path}")
        combined_model = tf.keras.models.load_model(model_checkpoint_path)
        
        if os.path.exists(epoch_file_path):
            with open(epoch_file_path, 'r') as f:
                try:
                    initial_epoch = int(f.read())
                    print(f"Resuming from epoch {initial_epoch + 1}.")
                except ValueError:
                    print("Warning: Could not read epoch number, starting from epoch 0.")
    else:
        print("Starting training from scratch.")
        combined_model = create_fomo_td_with_rnn_combined_model(
            input_sequence_shape=COMBINED_MODEL_INPUT_SHAPE,
            fomo_num_classes=1,
            motion_num_classes=NUM_MOTION_OUTPUTS,
            fomo_alpha=FOMO_ALPHA,
            fomo_backbone_cutoff_layer_name=FOMO_BACKBONE_CUTOFF_LAYER_NAME,
            rnn_type=config['rnn_type'],
            rnn_conv_lstm_filters=config['rnn_conv_lstm_filters'],
            rnn_gru_units=config['rnn_gru_units'],
            rnn_dense_units=config['rnn_dense_units']
        )
        
        try:
            fomo_processing_sub_model = combined_model.get_layer('time_distributed_fomo_processing').layer
            fomo_processing_sub_model.set_weights(fomo_stage1_model.get_weights())
            print("Successfully transferred weights from stage 1 FOMO model.")
        except Exception as e:
            print(f"Warning: Failed to transfer weights from stage 1 model: {e}")

        fomo_processing_sub_model.trainable = False
        print("FOMO sub-model frozen. Only the RNN head will be trained.")

        combined_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_STAGE2),
            loss={'motion_output': tf.keras.losses.MeanSquaredError()},
            metrics={'motion_output': [tf.keras.metrics.MeanAbsoluteError()]}
        )

    combined_model.summary()

    # Callback to save the epoch number
    class EpochSaverCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            with open(epoch_file_path, 'w') as f:
                f.write(str(epoch))

    # Set up callbacks
    monitor_metric = 'val_loss' if val_gen and val_gen.sequences_data else 'loss'
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, save_best_only=True, monitor=monitor_metric),
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_metric, factor=0.2, patience=5, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor=monitor_metric, patience=10, restore_best_weights=True),
        tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1, update_freq='epoch'),
        EpochSaverCallback()
    ]

    # Train the model
    print(f"\nStarting/resuming training for experiment '{config['name']}'...")
    try:
        combined_model.fit(
            train_gen,
            epochs=EPOCHS_STAGE2,
            validation_data=val_gen if val_gen and val_gen.sequences_data else None,
            callbacks=callbacks,
            initial_epoch=initial_epoch
        )
        print(f"\nTraining for experiment '{config['name']}' finished.")
    except Exception as e:
        print(f"\nAn error occurred during training for experiment '{config['name']}': {e}")

def main_stage2_training():
    print("Phase 2: Training the RNN head of the combined model.")

    if not os.path.exists(STAGE1_FOMO_MODEL_PATH):
        print(f"Error: Stage 1 FOMO model not found at {STAGE1_FOMO_MODEL_PATH}")
        return

    fomo_stage1_model = tf.keras.models.load_model(
        STAGE1_FOMO_MODEL_PATH,
        custom_objects={'loss': fomo_loss_function(num_classes_with_background=NUM_CLASSES_MODEL_OUTPUT_FOMO)}
    )
    print("Stage 1 FOMO model loaded for weight transfer.")
    video_files = sorted(glob.glob(os.path.join(VIDEO_DATA_DIR, '*.mp4')))[:50] # DEBUG: limit files

    if not video_files:
        print(f"No videos found in {VIDEO_DATA_DIR}.")
        return
    if not os.path.isdir(ANNOTATION_DATA_DIR):
        print(f"Error: Annotation directory not found at {ANNOTATION_DATA_DIR}")
        return

    val_split_idx = int(len(video_files) * 0.8)
    train_video_files = video_files[:val_split_idx]
    val_video_files = video_files[val_split_idx:]
    print(f"Using {len(train_video_files)} videos for training and {len(val_video_files)} for validation.")

    train_seq_generator = VideoSequenceDataGenerator(
        video_files=train_video_files,
        annotation_dir=ANNOTATION_DATA_DIR,
        batch_size=BATCH_SIZE_STAGE2,
        sequence_length=SEQUENCE_LENGTH,
        input_shape=INPUT_SHAPE,
        shuffle=True
    )

    validation_seq_generator = None
    if val_video_files:
        validation_seq_generator = VideoSequenceDataGenerator(
            video_files=val_video_files,
            annotation_dir=ANNOTATION_DATA_DIR,
            batch_size=BATCH_SIZE_STAGE2,
            sequence_length=SEQUENCE_LENGTH,
            input_shape=INPUT_SHAPE,
            shuffle=False
        )

    if not train_seq_generator.sequences_data:
        print("Error: No training data could be loaded. Check video and CSV files.")
        return

    # Run all defined experiments
    for exp_config in EXPERIMENTS:
        run_experiment(exp_config, fomo_stage1_model, train_seq_generator, validation_seq_generator)

if __name__ == '__main__':
    main_stage2_training()
