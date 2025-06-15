import os
import glob
import cv2
import numpy as np
import tensorflow as tf
import math
import json
from tensorflow.keras.utils import Sequence as KerasSequence # Renommer pour éviter conflit si on importe Sequence de typing
from joblib import Memory

memory = Memory("./cachedir", verbose=0) # Pour la mise en cache des séquences extraites

from fomo_trainer import (
    create_fomo_rnn_combined_model, 
    fomo_loss_function, # Pour charger le modèle FOMO de la phase 1
    create_fomo_model # Pour charger le modèle FOMO de la phase 1
)

# --- Configuration (similaire à simulate_person_detector.py et train_person_detector.py) ---
VIDEO_DATA_DIR = 'data/videos_for_rnn_training' # Répertoire contenant les vidéos pour la phase 2
STAGE1_FOMO_MODEL_PATH = 'person_detector_fomo.h5' # Modèle FOMO de la phase 1
COMBINED_MODEL_SAVE_PATH = 'fomo_rnn_combined_model_stage2.h5' # Modèle combiné final

INPUT_HEIGHT = 96
INPUT_WIDTH = 96
INPUT_CHANNELS = 3
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS) # Pour une image unique
SEQUENCE_LENGTH = 10  # Nombre d'images par séquence pour le RNN
COMBINED_MODEL_INPUT_SHAPE = (SEQUENCE_LENGTH, INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)

# Paramètres FOMO (doivent correspondre au modèle de phase 1 et à la config du modèle combiné)
FOMO_ALPHA = 0.35
# Assurez-vous que ce BACKBONE_CUTOFF_LAYER_NAME correspond à celui utilisé pour STAGE1_FOMO_MODEL_PATH
# et à celui que vous voulez dans le modèle combiné.
# Le train_person_detector.py utilise 'block_3_expand_relu' (réduction par 4)
# Le simulate_person_detector.py utilise SPATIAL_REDUCTION = 4
# Le fomo_trainer.py dans create_fomo_rnn_combined_model a 'block_6_expand_relu' par défaut.
# Il est crucial que ces éléments soient cohérents.
# Pour cet exemple, nous allons supposer que le modèle de phase 1 et le combiné utilisent la même config.
_known_cutoffs_local = {
    'expanded_conv_project_BN': 2,
    'block_3_expand_relu': 4,
    'block_6_expand_relu': 8,
    'block_10_expand_relu': 16
}
FOMO_BACKBONE_CUTOFF_LAYER_NAME = 'block_3_expand_relu' # Doit correspondre à la phase 1
if FOMO_BACKBONE_CUTOFF_LAYER_NAME not in _known_cutoffs_local:
    raise ValueError(f"Cutoff layer {FOMO_BACKBONE_CUTOFF_LAYER_NAME} non supporté ici.")
SPATIAL_REDUCTION = _known_cutoffs_local[FOMO_BACKBONE_CUTOFF_LAYER_NAME]

GRID_HEIGHT = INPUT_HEIGHT // SPATIAL_REDUCTION
GRID_WIDTH = INPUT_WIDTH // SPATIAL_REDUCTION
PERSON_CLASS_ID = 0 
NUM_OBJECT_CLASSES_FOMO = 1 # 'personne'
NUM_CLASSES_MODEL_OUTPUT_FOMO = NUM_OBJECT_CLASSES_FOMO + 1
DETECTION_THRESHOLD = 0.5 # Seuil pour considérer une détection lors de la génération d'étiquettes

# Paramètres pour la génération d'étiquettes de mouvement (auto-supervision)
HIST_NUM_BINS = 10
HIST_RANGE = (-INPUT_WIDTH // 2, INPUT_WIDTH // 2)
MOTION_HIST_ZERO_THRESHOLD = INPUT_WIDTH / 20
PAN_PREDICTION_HIST_DIFF_THRESHOLD = HIST_NUM_BINS // 5
MOTION_CLASSES = ["Tourner Gauche", "Rester Statique", "Tourner Droite"]
NUM_MOTION_CLASSES = len(MOTION_CLASSES)

# Hyperparamètres d'entraînement pour la phase 2
BATCH_SIZE_STAGE2 = 8
LEARNING_RATE_STAGE2 = 0.0005
EPOCHS_STAGE2 = 30
# --- Fin de la Configuration ---

# Fonctions utilitaires (adaptées de simulate_person_detector.py)
def preprocess_single_frame(frame, target_shape):
    img_resized = cv2.resize(frame, (target_shape[1], target_shape[0]))
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized

def postprocess_fomo_heatmap(heatmap_output, original_frame_shape, grid_shape, person_class_id, threshold):
    detected_centroids = []
    grid_h, grid_w = grid_shape
    frame_h, frame_w = original_frame_shape
    if heatmap_output.shape[-1] > 1:
        heatmap_probs = tf.nn.softmax(heatmap_output, axis=-1).numpy()
    else:
        heatmap_probs = tf.math.sigmoid(heatmap_output).numpy()
    
    person_heatmap = heatmap_probs[0, :, :, person_class_id]
    for r in range(grid_h):
        for c in range(grid_w):
            confidence = person_heatmap[r, c]
            if confidence > threshold:
                center_x_norm = (c + 0.5) / grid_w
                center_y_norm = (r + 0.5) / grid_h
                center_x_orig = int(center_x_norm * frame_w)
                center_y_orig = int(center_y_norm * frame_h)
                detected_centroids.append((center_x_orig, center_y_orig, confidence, r, c))
    return detected_centroids

def compute_displacements_for_sequence(frame_sequence_centroids):
    """ Calcule les déplacements horizontaux pour une séquence de listes de centroïdes. """
    all_horizontal_displacements = []
    for i in range(len(frame_sequence_centroids) - 1):
        prev_centroids = frame_sequence_centroids[i]
        current_centroids = frame_sequence_centroids[i+1]
        
        # Adapter la logique de compute_optical_flow de simulate_person_detector.py
        # Pour simplifier ici, on va juste prendre les déplacements bruts si des objets sont présents.
        # Une version plus robuste utiliserait l'appariement par grille.
        
        # Convertir en map pour la logique d'appariement par grille (si utilisée)
        # prev_grid_map = {(r, c): (x, y, conf) for x, y, conf, r, c in prev_centroids}
        # current_grid_map = {(r, c): (x, y, conf) for x, y, conf, r, c in current_centroids}
        # ... (logique d'appariement de compute_optical_flow) ...
        # Pour cet exemple, on va faire une version simplifiée :
        # Si des centroïdes existent dans les deux, on prend la moyenne du mouvement.
        if prev_centroids and current_centroids:
            # Simplification: on prend la différence des moyennes des positions x
            # Une vraie implémentation devrait apparier les objets.
            avg_prev_x = np.mean([c[0] for c in prev_centroids])
            avg_curr_x = np.mean([c[0] for c in current_centroids])
            all_horizontal_displacements.append(avg_curr_x - avg_prev_x)

    return all_horizontal_displacements


def generate_motion_label(horizontal_displacements_sequence):
    """ Génère une étiquette de mouvement basée sur une séquence de déplacements. """
    if not horizontal_displacements_sequence:
        return MOTION_CLASSES.index("Rester Statique") # Statique par défaut

    hist_values, bin_edges = np.histogram(horizontal_displacements_sequence, bins=HIST_NUM_BINS, range=HIST_RANGE)
    
    left_motion_sum = 0
    right_motion_sum = 0
    for i in range(len(hist_values)):
        bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
        if bin_center < -MOTION_HIST_ZERO_THRESHOLD:
            left_motion_sum += hist_values[i]
        elif bin_center > MOTION_HIST_ZERO_THRESHOLD:
            right_motion_sum += hist_values[i]

    if right_motion_sum > left_motion_sum + PAN_PREDICTION_HIST_DIFF_THRESHOLD:
        return MOTION_CLASSES.index("Tourner Gauche") # Objets vont à droite -> caméra à gauche
    elif left_motion_sum > right_motion_sum + PAN_PREDICTION_HIST_DIFF_THRESHOLD:
        return MOTION_CLASSES.index("Tourner Droite") # Objets vont à gauche -> caméra à droite
    else:
        return MOTION_CLASSES.index("Rester Statique")


class VideoSequenceDataGenerator(KerasSequence):
    def __init__(self, video_files, batch_size, sequence_length, input_shape, 
                 stage1_fomo_model, grid_shape, person_class_id, detection_threshold,
                 motion_label_params, shuffle=True):
        self.video_files = video_files
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.input_shape = input_shape # Shape d'une image unique
        self.stage1_fomo_model = stage1_fomo_model
        self.grid_shape = grid_shape
        self.person_class_id = person_class_id
        self.detection_threshold = detection_threshold
        self.motion_label_params = motion_label_params # dict avec HIST_NUM_BINS, etc.
        self.shuffle = shuffle

        self.sequences_data = self._extract_sequences_from_videos()
        self.indexes = np.arange(len(self.sequences_data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    @memory.cache()
    def _extract_sequences_from_videos(self):
        all_sequences_with_labels = []
        print("Extraction des séquences et génération des étiquettes auto-supervisées...")
        for video_path in self.video_files:
            cap = cv2.VideoCapture(video_path)
            frames_buffer = []
            centroids_buffer = [] # Stocke les centroïdes pour chaque frame dans frames_buffer

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                original_h, original_w = frame.shape[:2]
                processed_frame = preprocess_single_frame(frame, self.input_shape)
                frames_buffer.append(processed_frame)

                # Obtenir les centroïdes avec le modèle FOMO de phase 1
                input_tensor_fomo = np.expand_dims(processed_frame, axis=0)
                heatmap_fomo = self.stage1_fomo_model.predict(input_tensor_fomo)
                current_centroids = postprocess_fomo_heatmap(
                    heatmap_fomo, (original_h, original_w), self.grid_shape, 
                    self.person_class_id, self.detection_threshold
                )
                centroids_buffer.append(current_centroids)

                if len(frames_buffer) == self.sequence_length:
                    # Calculer les déplacements pour la séquence de centroïdes actuelle
                    displacements = compute_displacements_for_sequence(centroids_buffer)
                    motion_label_idx = generate_motion_label(displacements)
                    
                    # Convertir en one-hot pour la loss categorical_crossentropy
                    motion_label_one_hot = tf.keras.utils.to_categorical(motion_label_idx, num_classes=len(MOTION_CLASSES))
                    
                    all_sequences_with_labels.append((list(frames_buffer), motion_label_one_hot))
                    
                    # Faire glisser la fenêtre
                    frames_buffer.pop(0)
                    centroids_buffer.pop(0)
            cap.release()
        print(f"Extraction terminée. {len(all_sequences_with_labels)} séquences générées.")
        return all_sequences_with_labels

    def __len__(self):
        return math.ceil(len(self.sequences_data) / self.batch_size)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_sequences = []
        batch_motion_labels = []

        for i in batch_indexes:
            sequence_frames, motion_label = self.sequences_data[i]
            batch_sequences.append(np.array(sequence_frames))
            batch_motion_labels.append(motion_label)
            
        # Pour le modèle combiné, la sortie FOMO n'est pas directement supervisée dans cette phase 2
        # si on se concentre uniquement sur l'entraînement de la tête RNN.
        # On pourrait fournir des "dummy" labels pour la sortie FOMO si la compilation l'exige.
        # Ici, on suppose que la compilation du modèle combiné pour la phase 2
        # ne spécifiera une perte que pour la sortie 'motion_output'.
        return np.array(batch_sequences), np.array(batch_motion_labels)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def main_stage2_training():
    print("Phase 2: Entraînement de la tête RNN du modèle combiné.")

    # 1. Charger le modèle FOMO de la phase 1 (pour la génération d'étiquettes)
    if not os.path.exists(STAGE1_FOMO_MODEL_PATH):
        print(f"Erreur: Modèle FOMO de Phase 1 non trouvé à {STAGE1_FOMO_MODEL_PATH}")
        return
    
    # Le modèle FOMO de phase 1 a été sauvegardé après compilation avec fomo_loss_function.
    # Il faut donc le fournir lors du chargement.
    fomo_stage1_model = tf.keras.models.load_model(
        STAGE1_FOMO_MODEL_PATH,
        custom_objects={'loss': fomo_loss_function(num_classes_with_background=NUM_CLASSES_MODEL_OUTPUT_FOMO)}
    )
    print("Modèle FOMO de Phase 1 chargé.")

    # 2. Préparer le générateur de données pour la phase 2
    video_files = glob.glob(os.path.join(VIDEO_DATA_DIR, '*.mp4')) # ou autres formats
    if not video_files:
        print(f"Aucune vidéo trouvée dans {VIDEO_DATA_DIR} pour l'entraînement de la phase 2.")
        return

    motion_label_params = {
        'HIST_NUM_BINS': HIST_NUM_BINS,
        'HIST_RANGE': HIST_RANGE,
        'MOTION_HIST_ZERO_THRESHOLD': MOTION_HIST_ZERO_THRESHOLD,
        'PAN_PREDICTION_HIST_DIFF_THRESHOLD': PAN_PREDICTION_HIST_DIFF_THRESHOLD
    }

    train_seq_generator = VideoSequenceDataGenerator(
        video_files=video_files, # Diviser en train/val si nécessaire
        batch_size=BATCH_SIZE_STAGE2,
        sequence_length=SEQUENCE_LENGTH,
        input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS),
        stage1_fomo_model=fomo_stage1_model,
        grid_shape=(GRID_HEIGHT, GRID_WIDTH),
        person_class_id=PERSON_CLASS_ID,
        detection_threshold=DETECTION_THRESHOLD,
        motion_label_params=motion_label_params
    )
    
    # (Optionnel) Créer un validation_seq_generator si vous avez un ensemble de validation de vidéos

    # 3. Créer le modèle combiné FOMO-RNN
    combined_model = create_fomo_rnn_combined_model(
        input_sequence_shape=COMBINED_MODEL_INPUT_SHAPE,
        fomo_num_classes=NUM_OBJECT_CLASSES_FOMO,
        motion_num_classes=NUM_MOTION_CLASSES,
        fomo_alpha=FOMO_ALPHA,
        fomo_backbone_cutoff_layer_name=FOMO_BACKBONE_CUTOFF_LAYER_NAME,
        # Les autres paramètres de create_fomo_rnn_combined_model peuvent être ajustés
    )
    print("Modèle combiné FOMO-RNN créé.")
    combined_model.summary()

    # 4. Transférer les poids du backbone FOMO
    # Le modèle FOMO de phase 1 est 'fomo_stage1_model'
    # Le backbone dans 'combined_model' est nommé 'fomo_backbone' (c'est un sous-modèle)
    
    # Option A: Si fomo_stage1_model est juste le backbone + tête FOMO simple
    # et que fomo_backbone dans combined_model est architecturalement identique au backbone de fomo_stage1_model
    print("Tentative de transfert des poids du backbone...")
    try:
        # Le modèle FOMO de phase 1 (créé par create_fomo_model) a son backbone directement.
        # Le modèle combiné a un sous-modèle nommé 'fomo_backbone'.
        # Nous devons extraire les poids du backbone du modèle de phase 1.
        
        # Créons une instance du backbone seul à partir de create_fomo_model pour obtenir ses noms de couches
        # ou chargeons les poids couche par couche par nom.
        # Plus simple: si fomo_stage1_model est le modèle complet de la phase 1,
        # et que son backbone est compatible avec le sous-modèle 'fomo_backbone' du modèle combiné.
        
        # Le modèle fomo_stage1_model est le modèle complet (backbone + tête fomo).
        # Le modèle combiné a un sous-modèle 'fomo_backbone'.
        # On peut copier les poids des couches correspondantes par nom.
        
        # Le plus simple est de s'assurer que le backbone du modèle de phase 1
        # (jusqu'à FOMO_BACKBONE_CUTOFF_LAYER_NAME) a les mêmes noms de couches
        # que le sous-modèle 'fomo_backbone' dans le modèle combiné.
        # tf.keras.applications.MobileNetV2 est utilisé dans les deux cas.
        
        # Accéder au modèle fomo_backbone encapsulé dans la couche TimeDistributed
        time_distributed_layer = combined_model.get_layer('time_distributed_backbone')
        fomo_backbone_combined = time_distributed_layer.layer # C'est le modèle fomo_backbone
        
        num_layers_transferred = 0
        for layer_stage1 in fomo_stage1_model.layers:
            if layer_stage1.name in [l.name for l in fomo_backbone_combined.layers]:
                target_layer_combined = fomo_backbone_combined.get_layer(layer_stage1.name)
                if target_layer_combined.get_weights(): # S'assurer que la couche a des poids
                    target_layer_combined.set_weights(layer_stage1.get_weights())
                    num_layers_transferred +=1
        print(f"{num_layers_transferred} couches de poids transférées au backbone du modèle combiné.")

        # Transférer aussi les poids de la tête FOMO si elle est identique
        # La tête FOMO du modèle combiné a les couches 'fomo_head_conv' et 'fomo_output'
        # Celles du modèle de phase 1 sont les dernières couches après le backbone_cutoff.
        # Supposons que les noms correspondent ou que la structure est la même.
        # Pour l'instant, concentrons-nous sur le backbone.

    except Exception as e:
        print(f"Avertissement: Erreur lors du transfert des poids du backbone: {e}")
        print("L'entraînement de la phase 2 commencera avec un backbone initialisé aléatoirement (ou pré-entraîné ImageNet).")


    # 5. Geler les couches du backbone et de la tête FOMO
    print("Gel des couches du backbone et de la tête FOMO...")
    # fomo_backbone_combined est déjà défini correctement ci-dessus pour le transfert de poids
    # et pointe vers le modèle fomo_backbone encapsulé.
    fomo_backbone_combined.trainable = False
    
    # Geler aussi les couches de la tête FOMO du modèle combiné
    combined_model.get_layer('fomo_head_conv').trainable = False
    combined_model.get_layer('fomo_output').trainable = False # La sortie FOMO
    
    print("Couches gelées. Seule la tête RNN sera entraînée initialement.")
    # Vérifier le statut des couches entraînables
    # for layer in combined_model.layers:
    #     print(f"Layer: {layer.name}, Trainable: {layer.trainable}")
    # combined_model.summary() # Pour voir les paramètres entraînables

    # 6. Compiler le modèle combiné pour l'entraînement de la tête RNN
    # Nous ne fournissons une perte que pour la sortie 'motion_output'.
    # Pour la sortie 'fomo_output' qui est gelée, nous pouvons spécifier None comme perte.
    combined_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_STAGE2),
        loss={
            'fomo_output': None,  # Pas de perte calculée pour la sortie FOMO gelée
            'motion_output': 'categorical_crossentropy'
        },
        metrics={'motion_output': ['accuracy']} # Métriques uniquement pour la sortie de mouvement
    )

    # 7. Entraîner le modèle (principalement la tête RNN)
    print("Début de l'entraînement de la phase 2 (tête RNN)...")
    try:
        combined_model.fit(
            train_seq_generator,
            epochs=EPOCHS_STAGE2,
            # validation_data=validation_seq_generator, # Si vous en avez un
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(COMBINED_MODEL_SAVE_PATH, save_best_only=True, monitor='loss'), # ou 'val_loss'
                tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.00001),
                tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7, restore_best_weights=True)
            ]
        )
        print("Entraînement de la phase 2 terminé.")
        print(f"Modèle combiné sauvegardé dans {COMBINED_MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Erreur durant l'entraînement de la phase 2: {e}")

    # Optionnel: Phase de Fine-tuning
    # Dégelez quelques couches supérieures du backbone et ré-entraînez avec un learning rate plus faible.
    # print("Début du fine-tuning du modèle combiné...")
    # fomo_backbone_combined.trainable = True # Dégeler tout le backbone
    # # Ou dégeler sélectivement les dernières couches du backbone
    # # for layer in fomo_backbone_combined.layers[-N:]: # Dégeler les N dernières couches
    # #    layer.trainable = True
    #
    # combined_model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_STAGE2 / 10), # LR plus faible
    #     loss={'motion_output': 'categorical_crossentropy'},
    #     metrics={'motion_output': ['accuracy']}
    # )
    # combined_model.fit(...) # Entraîner pour quelques époques de plus


if __name__ == '__main__':
    main_stage2_training()
