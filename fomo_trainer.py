import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import math

# Dictionary of known MobileNetV2 layer names for backbone cutoff and their corresponding spatial reduction factors
_known_cutoffs = {
    'expanded_conv_project_BN': 2,   # Results in 1/2 spatial reduction (e.g., 96x96 -> 48x48)
    'block_3_expand_relu': 4,       # Results in 1/4 spatial reduction (e.g., 96x96 -> 24x24)
    'block_6_expand_relu': 8,       # Results in 1/8 spatial reduction (e.g., 96x96 -> 12x12)
    'block_10_expand_relu': 16      # Results in 1/16 spatial reduction (e.g., 96x96 -> 6x6)
}

def create_fomo_model(input_shape, num_classes, alpha=0.35, backbone_cutoff_layer_name='block_6_expand_relu', head_conv_filters=32):
    """
    Creates a FOMO model based on MobileNetV2.

    Args:
        input_shape (tuple): Shape of the input image (height, width, channels).
        num_classes (int): Number of object classes (excluding background).
        alpha (float): Alpha factor for MobileNetV2 (controls network width).
        backbone_cutoff_layer_name (str): Name of the MobileNetV2 layer where the backbone is cut.
                                          'block_6_expand_relu' corresponds to an 8x spatial reduction.
        head_conv_filters (int): Number of filters in the first Conv2D layer of the FOMO head.

    Returns:
        tf.keras.Model: The compiled FOMO model.
    """
    num_classes_with_background = num_classes + 1

    # Load MobileNetV2 pre-trained on ImageNet, without the top classification layer
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        alpha=alpha,
        include_top=False,
        weights='imagenet'
    )

    # Extract the output at the specified cutoff layer
    try:
        fomo_backbone_output = backbone.get_layer(backbone_cutoff_layer_name).output
    except ValueError:
        print(f"Error: Layer '{backbone_cutoff_layer_name}' not found. Check the layer name for MobileNetV2 alpha {alpha}.")
        print("Available layers in the backbone:")
        for layer in backbone.layers:
            print(layer.name)
        raise

    # FOMO Head
    # The head is applied in a fully convolutional manner to the backbone output
    # First 1x1 convolutional layer to reduce dimensionality or transform features
    head = layers.Conv2D(filters=head_conv_filters, kernel_size=(1, 1), padding='same', activation='relu')(fomo_backbone_output)
    # Logits layer: a 1x1 convolutional layer to produce scores for each class per grid cell
    # No activation here as logits are used for the loss function (softmax will be applied in the loss)
    logits = layers.Conv2D(filters=num_classes_with_background, kernel_size=(1, 1), padding='same', activation=None)(head)

    model = tf.keras.Model(inputs=backbone.input, outputs=logits)
    return model

def fomo_loss_function(object_weight=100.0, num_classes_with_background=None):
    """
    Creates the loss function for FOMO.

    Args:
        object_weight (float): Weight to apply to cells containing an object.
        num_classes_with_background (int): Total number of classes including background.
                                           The background class index is assumed to be num_classes_with_background - 1.

    Returns:
        function: The loss function to use with Keras.
    """
    if num_classes_with_background is None:
        raise ValueError("num_classes_with_background must be specified.")

    background_class_idx = num_classes_with_background - 1

    def loss(y_true, y_pred):
        """
        Args:
            y_true (tf.Tensor): Target heatmap. Shape: (batch, grid_h, grid_w, num_classes_with_background). One-hot encoded.
            y_pred (tf.Tensor): Predicted heatmap (logits). Shape: (batch, grid_h, grid_w, num_classes_with_background).
        """
        # Calculate softmax cross-entropy loss per pixel (cell)
        # from_logits=True because y_pred are logits
        per_cell_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)

        # Create weights
        # Identify object vs background cells from y_true
        # y_true is one-hot. If the background class is not activated, it's an object.
        is_object_cell = tf.cast(tf.logical_not(tf.equal(tf.argmax(y_true, axis=-1), background_class_idx)), dtype=tf.float32)
        
        # Apply weights: object_weight for object cells, 1.0 for background cells
        weights = is_object_cell * (object_weight - 1.0) + 1.0
        
        weighted_loss = per_cell_loss * weights
        
        # Sum losses over grid cells and take the mean over the batch
        # Or sum over all dimensions and divide by batch size for per-image loss
        # Or sum over all dimensions and divide by total number of elements for per-cell loss
        # Edge Impulse typically normalizes by the number of cells in the batch.
        # Here, we will take the mean of weighted per-cell losses across the entire batch.
        return tf.reduce_mean(weighted_loss)

    return loss


class FomoDataGenerator(tf.keras.utils.Sequence):
    """
    Data generator for FOMO training.
    Loads images and prepares target heatmaps from centroid annotations.
    """
    def __init__(self, image_filenames, annotations, batch_size, input_shape, grid_shape, num_classes, shuffle=True):
        """
        Args:
            image_filenames (list): List of paths to image files.
            annotations (list): List of annotations. Each annotation is a list of centroids for an image.
                                 A centroid is a tuple (x_center_norm, y_center_norm, class_id).
                                 x_center_norm, y_center_norm are normalized (0-1).
                                 class_id is the object class ID (0 to num_classes-1).
            batch_size (int): Batch size.
            input_shape (tuple): Shape of the input image (height, width, channels).
            grid_shape (tuple): Shape of the output grid (grid_h, grid_w).
            num_classes (int): Number of object classes (excluding background).
            shuffle (bool): If true, shuffle data at each epoch.
        """
        self.image_filenames = image_filenames
        self.annotations = annotations
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.grid_shape = grid_shape
        self.num_classes = num_classes
        self.num_classes_with_background = num_classes + 1
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.image_filenames))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.image_filenames) / self.batch_size)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_images = []
        batch_heatmaps = []

        for i in batch_indexes:
            # Load image
            img = tf.keras.preprocessing.image.load_img(
                self.image_filenames[i], 
                target_size=(self.input_shape[0], self.input_shape[1]),
                color_mode='rgb' # Ensure 3 channels for MobileNetV2
            )
            img = tf.keras.preprocessing.image.img_to_array(img)
            img = img / 255.0  # Normalize to [0, 1]
            batch_images.append(img)
            
            # Prepare target heatmap
            heatmap = self._prepare_target_heatmap(self.annotations[i], self.grid_shape, self.num_classes)
            batch_heatmaps.append(heatmap)
            
        return np.array(batch_images), np.array(batch_heatmaps)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _prepare_target_heatmap(self, item_annotations, grid_shape, num_classes):
        """
        Prepares the target heatmap for a single image.

        Args:
            item_annotations (list): List of centroids (x_center_norm, y_center_norm, class_id) for the image.
            grid_shape (tuple): Grid shape (grid_h, grid_w).
            num_classes (int): Number of object classes.

        Returns:
            np.array: One-hot encoded target heatmap. Shape (grid_h, grid_w, num_classes + 1).
        """
        grid_h, grid_w = grid_shape
        num_classes_with_background = num_classes + 1
        background_class_idx = num_classes # Background class index

        heatmap = np.zeros((grid_h, grid_w, num_classes_with_background), dtype=np.float32)
        # Initialize all cells to the background class
        heatmap[:, :, background_class_idx] = 1.0

        for ann in item_annotations:
            x_center_norm, y_center_norm, class_id = ann
            
            # Convert normalized coordinates to grid coordinates
            gx = int(x_center_norm * grid_w)
            gy = int(y_center_norm * grid_h)

            # Ensure coordinates are within grid boundaries
            gx = min(max(gx, 0), grid_w - 1)
            gy = min(max(gy, 0), grid_h - 1)

            if class_id < 0 or class_id >= num_classes:
                print(f"Warning: class_id {class_id} out of bounds [0, {num_classes-1}]. Ignored.")
                continue

            # Set the object class for this cell
            heatmap[gy, gx, class_id] = 1.0
            # Reset the background class for this cell
            heatmap[gy, gx, background_class_idx] = 0.0
            
        return heatmap


if __name__ == '__main__':
    # Example parameters
    INPUT_HEIGHT = 96
    INPUT_WIDTH = 96
    INPUT_CHANNELS = 3  # MobileNetV2 expects 3 channels. If grayscale input, convert it.
    INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)
    
    NUM_CLASSES = 2  # E.g.: "beer", "can"
    # The background class will be added automatically, so num_classes_with_background = NUM_CLASSES + 1

    # MobileNetV2 alpha. 0.35 is the smallest standard alpha in tf.keras.applications.
    # Smaller alphas like 0.1 mentioned in Edge Impulse docs might require a custom model.
    ALPHA = 0.35 

    # The backbone output is 8x smaller than the input with 'block_6_expand_relu'
    # Example: Choose a cutoff layer. This can be changed to test different configurations.
    BACKBONE_CUTOFF_LAYER_NAME = 'block_6_expand_relu' 

    if BACKBONE_CUTOFF_LAYER_NAME not in _known_cutoffs:
        raise ValueError(
            f"Unsupported BACKBONE_CUTOFF_LAYER_NAME in fomo_trainer example: {BACKBONE_CUTOFF_LAYER_NAME}. "
            f"Supported are: {list(_known_cutoffs.keys())}"
        )
    SPATIAL_REDUCTION = _known_cutoffs[BACKBONE_CUTOFF_LAYER_NAME]
    
    GRID_HEIGHT = INPUT_HEIGHT // SPATIAL_REDUCTION
    GRID_WIDTH = INPUT_WIDTH // SPATIAL_REDUCTION
    GRID_SHAPE = (GRID_HEIGHT, GRID_WIDTH)

    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    EPOCHS = 50
    OBJECT_WEIGHT = 100.0 # Weight for cells containing an object in the loss function

    # Creating dummy data for the example
    num_samples = 100
    dummy_image_filenames = [f"image_{i}.png" for i in range(num_samples)]
    # Dummy annotations: list of lists. Each sublist contains centroids for an image.
    # A centroid: (x_center_norm, y_center_norm, class_id)
    dummy_annotations = []
    for _ in range(num_samples):
        num_objects_in_image = np.random.randint(0, 5)
        sample_annots = []
        for _ in range(num_objects_in_image):
            x_norm, y_norm = np.random.rand(), np.random.rand()
            class_id = np.random.randint(0, NUM_CLASSES)
            sample_annots.append((x_norm, y_norm, class_id))
        dummy_annotations.append(sample_annots)

    # Create the data generator
    train_generator = FomoDataGenerator(
        image_filenames=dummy_image_filenames,
        annotations=dummy_annotations,
        batch_size=BATCH_SIZE,
        input_shape=INPUT_SHAPE,
        grid_shape=GRID_SHAPE,
        num_classes=NUM_CLASSES
    )

    # Create the FOMO model
    # Note: num_classes is the number of object classes. The background class is handled internally.
    model = create_fomo_model(
        INPUT_SHAPE, 
        NUM_CLASSES, 
        alpha=ALPHA,
        backbone_cutoff_layer_name=BACKBONE_CUTOFF_LAYER_NAME
    )
    model.summary()

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss_fn = fomo_loss_function(object_weight=OBJECT_WEIGHT, num_classes_with_background=NUM_CLASSES + 1)
    
    model.compile(optimizer=optimizer, loss=loss_fn)

    print(f"\nStarting training with dummy data...")
    print(f"Input shape: {INPUT_SHAPE}, Grid shape: {GRID_SHAPE}, Num object classes: {NUM_CLASSES}")
    
    # Train the model
    # Note: For actual training, you would need validation data.
    # model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator)
    # The following model.fit call is for testing the components with dummy data.
    # However, FomoDataGenerator now expects real image file paths.
    # Running this directly will cause a FileNotFoundError with dummy filenames.
    # To train the actual person detector, run train_person_detector.py.
    print("\nSkipping model.fit() in fomo_trainer.py's main block as it uses dummy file names.")
    print("To train the actual person detector, please run train_person_detector.py.")
    # try:
    #     # model.fit(train_generator, epochs=1) # Train for 1 epoch for a quick test
    #     print("\nTraining (quick test) completed.")
    #     print("For actual training, load real data and increase the number of epochs.")
    # except Exception as e:
    #     print(f"\nAn error occurred during test training: {e}")
    #     print("This might be due to configuration issues or dependencies.")

    # Next steps:
    # 1. Implement actual image loading in FomoDataGenerator.
    # 2. Prepare your own datasets of images and centroid annotations.
    # 3. Adjust hyperparameters (learning rate, batch size, alpha, object_weight, etc.).
    # 4. Train the model on your data.
    # 5. Evaluate the model and export it for inference.


def create_camera_motion_rnn_model(sequence_length, num_features, num_motion_classes=3, rnn_units=32, dense_units=16):
    """
    Crée un modèle RNN simple pour prédire le mouvement de la caméra.

    Args:
        sequence_length (int): Nombre de pas de temps dans la séquence d'entrée (par exemple, nombre d'histogrammes consécutifs).
        num_features (int): Nombre de caractéristiques à chaque pas de temps (par exemple, HIST_NUM_BINS).
        num_motion_classes (int): Nombre de classes de mouvement de caméra à prédire (par exemple, 3 pour gauche, droite, statique).
        rnn_units (int): Nombre d'unités dans la couche RNN (LSTM/GRU).
        dense_units (int): Nombre d'unités dans la couche Dense intermédiaire.

    Returns:
        tf.keras.Model: Le modèle RNN compilé.
    """
    model = tf.keras.Sequential([
        layers.Input(shape=(sequence_length, num_features)),
        # Vous pouvez choisir LSTM ou GRU. GRU est souvent un peu plus rapide et performant sur des séquences plus courtes.
        layers.GRU(rnn_units, return_sequences=False), # return_sequences=False car on veut une sortie après toute la séquence
        layers.Dense(dense_units, activation='relu'),
        layers.Dropout(0.3), # Ajout d'un Dropout pour la régularisation
        layers.Dense(num_motion_classes, activation='softmax') # Sortie de classification pour les mouvements
    ])

    # Compiler le modèle est généralement fait avant l'entraînement.
    # Pour l'instant, nous le retournons non compilé ou avec un compilateur générique si besoin pour sauvegarde/chargement.
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
