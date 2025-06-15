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

memory = Memory("./cachedir", verbose=0) 

from fomo_trainer import (
    create_fomo_rnn_combined_model, # Sera utilisé si vous avez plusieurs types de modèles combinés
    create_fomo_td_with_rnn_combined_model, 
    fomo_loss_function, 
    create_fomo_model 
)

# --- Configuration ---
VIDEO_DATA_DIR = 'data/videos_for_rnn_training'
ANNOTATION_DATA_DIR = 'data/videos_for_rnn_training_annotations' # Répertoire pour les fichiers CSV d'annotations

VIDEO_DATA_DIR = '/Users/scampion/src/sport_video_scrapper/data/videos'
ANNOTATION_DATA_DIR = '/Users/scampion/src/sport_video_scrapper/camera_movement_reports' # Répertoire pour les fichiers CSV d'annotations



STAGE1_FOMO_MODEL_PATH = 'person_detector_fomo.h5'
COMBINED_MODEL_SAVE_PATH = 'fomo_td_rnn_regression_stage2.h5' # Nom de modèle mis à jour

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

GRID_HEIGHT = INPUT_HEIGHT // SPATIAL_REDUCTION # Utilisé par l'architecture FOMO
GRID_WIDTH = INPUT_WIDTH // SPATIAL_REDUCTION  # Utilisé par l'architecture FOMO
# PERSON_CLASS_ID, NUM_OBJECT_CLASSES_FOMO, DETECTION_THRESHOLD ne sont plus utilisés pour la génération d'étiquettes de mouvement
NUM_CLASSES_MODEL_OUTPUT_FOMO = 1 + 1 # Pour chargement modèle phase 1 (person + background)

# La sortie du RNN est maintenant une valeur de régression unique (mouvement X)
NUM_MOTION_OUTPUTS = 1 

# Hyperparamètres d'entraînement pour la phase 2
BATCH_SIZE_STAGE2 = 8
LEARNING_RATE_STAGE2 = 0.0005 # Peut nécessiter un ajustement pour la régression
EPOCHS_STAGE2 = 30 # Peut nécessiter un ajustement
RNN_TYPE_STAGE2 = 'convlstm' 
# --- Fin de la Configuration ---

def preprocess_single_frame(frame, target_shape):
    img_resized = cv2.resize(frame, (target_shape[1], target_shape[0]))
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized

# Les fonctions postprocess_fomo_heatmap, compute_displacements_for_sequence, 
# et generate_motion_label ne sont plus nécessaires ici car les étiquettes proviennent des CSV.
# Elles pourraient être conservées si utilisées ailleurs ou pour débogage, mais pour la clarté de cette modif, on les enlève de la portée directe du générateur.

class VideoSequenceDataGenerator(KerasSequence):
    def __init__(self, video_files, annotation_dir, batch_size, sequence_length, 
                 input_shape, shuffle=True):
        self.video_files = video_files
        self.annotation_dir = annotation_dir
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.input_shape = input_shape # Shape d'une image unique
        self.shuffle = shuffle

        self.sequences_data = self._extract_sequences_from_videos()
        self.indexes = np.arange(len(self.sequences_data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    #@memory.cache # Décommentez si vous souhaitez mettre en cache cette extraction coûteuse
    def _extract_sequences_from_videos(self):
        all_sequences_with_labels = []
        print("Extraction des séquences et des étiquettes de mouvement à partir des CSV...")
        for video_path in self.video_files:
            video_filename = os.path.basename(video_path)
            video_name_without_ext = os.path.splitext(video_filename)[0]
            annotation_path = os.path.join(self.annotation_dir, f"{video_name_without_ext}_movement.csv")

            if not os.path.exists(annotation_path):
                print(f"Attention: Fichier d'annotation CSV non trouvé pour {video_path} à {annotation_path}. Vidéo ignorée.")
                continue

            try:
                annotations_df = pd.read_csv(annotation_path)
                if 'Move_X' not in annotations_df.columns:
                    print(f"Attention: Colonne 'Move_X' non trouvée dans {annotation_path}. Vidéo ignorée.")
                    continue
                move_x_annotations = annotations_df['Move_X'].tolist()
            except Exception as e:
                print(f"Erreur lors de la lecture du fichier CSV {annotation_path}: {e}. Vidéo ignorée.")
                continue
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Erreur: Impossible d'ouvrir la vidéo {video_path}. Vidéo ignorée.")
                continue

            all_processed_frames_for_video = []
            frame_count_video = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                processed_frame = preprocess_single_frame(frame, self.input_shape)
                all_processed_frames_for_video.append(processed_frame)
                frame_count_video += 1
            cap.release()

            if frame_count_video != len(move_x_annotations):
                # Permettre une différence si le CSV a plus d'annotations que de frames (ex: annotations manuelles)
                # mais pas si le CSV en a moins.
                if len(move_x_annotations) < frame_count_video:
                    print(f"Attention: Moins d'annotations ({len(move_x_annotations)}) que de frames ({frame_count_video}) pour {video_path}. Vidéo ignorée.")
                    continue
                # Tronquer les annotations si elles sont plus longues que les frames disponibles
                move_x_annotations = move_x_annotations[:frame_count_video]


            num_possible_sequences = len(all_processed_frames_for_video) - self.sequence_length + 1
            if num_possible_sequences <= 0:
                print(f"Attention: Pas assez de frames ({len(all_processed_frames_for_video)}) pour former une séquence de longueur {self.sequence_length} pour {video_path}. Vidéo ignorée.")
                continue

            for i in range(num_possible_sequences):
                sequence_frames = all_processed_frames_for_video[i : i + self.sequence_length]
                # L'étiquette est le 'Move_X' de la dernière image de la séquence
                # S'assurer que l'index est valide pour move_x_annotations
                label_index = i + self.sequence_length - 1
                if label_index < len(move_x_annotations):
                    motion_label_value = move_x_annotations[label_index]
                    # S'assurer que la valeur est un float et non NaN avant de convertir
                    if pd.isna(motion_label_value):
                        print(f"Attention: Valeur NaN pour Move_X à l'index {label_index} pour la séquence commençant à {i} dans {video_path}. Séquence ignorée.")
                        continue
                    motion_label = np.array([float(motion_label_value)], dtype=np.float32)
                    all_sequences_with_labels.append((np.array(sequence_frames), motion_label))
                else:
                    # Ce cas ne devrait pas arriver avec la logique de troncature/vérification précédente
                    print(f"Attention: Index d'étiquette {label_index} hors limites pour les annotations de {video_path}. Séquence ignorée.")
        
        print(f"Extraction terminée. {len(all_sequences_with_labels)} séquences générées.")
        if not all_sequences_with_labels:
            # Ne pas lever d'erreur ici, permettre au script de continuer s'il y a plusieurs vidéos
            # et que certaines seulement ont des problèmes. L'erreur sera levée plus tard si aucune donnée.
            print("AVERTISSEMENT: Aucune séquence n'a pu être générée à partir des vidéos et annotations fournies.")
        return all_sequences_with_labels

    def __len__(self):
        return math.ceil(len(self.sequences_data) / self.batch_size)

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        batch_sequences = []
        batch_motion_labels = []

        for i in batch_indexes:
            sequence_frames, motion_label = self.sequences_data[i]
            batch_sequences.append(sequence_frames) # Déjà un np.array
            batch_motion_labels.append(motion_label) # Déjà un np.array([val])
            
        # La sortie 'motion_output' attend maintenant une forme (batch_size, 1)
        return np.array(batch_sequences), {'motion_output': np.array(batch_motion_labels)}

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def main_stage2_training():
    print("Phase 2: Entraînement de la tête RNN du modèle combiné.")

    # 1. Charger le modèle FOMO de la phase 1 (pour la génération d'étiquettes)
    if not os.path.exists(STAGE1_FOMO_MODEL_PATH):
        print(f"Erreur: Modèle FOMO de Phase 1 non trouvé à {STAGE1_FOMO_MODEL_PATH}")
        return
    
    # Le modèle FOMO de phase 1 est chargé pour le transfert de poids.
    fomo_stage1_model = tf.keras.models.load_model(
        STAGE1_FOMO_MODEL_PATH,
        custom_objects={'loss': fomo_loss_function(num_classes_with_background=NUM_CLASSES_MODEL_OUTPUT_FOMO)}
    )
    print("Modèle FOMO de Phase 1 chargé (pour transfert de poids).")

    # 2. Préparer le générateur de données pour la phase 2
    video_files = sorted(glob.glob(os.path.join(VIDEO_DATA_DIR, '*.mp4'))) # ou autres formats, triés pour la reproductibilité
    if not video_files:
        print(f"Aucune vidéo trouvée dans {VIDEO_DATA_DIR} pour l'entraînement de la phase 2.")
        return
    
    if not os.path.isdir(ANNOTATION_DATA_DIR):
        print(f"Erreur: Répertoire d'annotations CSV non trouvé à {ANNOTATION_DATA_DIR}")
        return

    # Diviser les fichiers vidéo pour un ensemble de validation simple (ex: 80% train, 20% val)
    # Assurez-vous d'avoir suffisamment de vidéos pour que cela ait un sens.
    num_videos = len(video_files)
    if num_videos < 5: # Arbitraire, mais il faut assez de données pour spliter
        print("Attention: Très peu de vidéos. Entraînement sans ensemble de validation dédié.")
        train_video_files = video_files
        val_video_files = []
    else:
        val_split_idx = int(num_videos * 0.8)
        train_video_files = video_files[:val_split_idx]
        val_video_files = video_files[val_split_idx:]
        print(f"Utilisation de {len(train_video_files)} vidéos pour l'entraînement et {len(val_video_files)} pour la validation.")


    train_seq_generator = VideoSequenceDataGenerator(
        video_files=train_video_files, 
        annotation_dir=ANNOTATION_DATA_DIR,
        batch_size=BATCH_SIZE_STAGE2,
        sequence_length=SEQUENCE_LENGTH,
        input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS),
        shuffle=True
    )
    
    validation_seq_generator = None
    if val_video_files:
        validation_seq_generator = VideoSequenceDataGenerator(
            video_files=val_video_files,
            annotation_dir=ANNOTATION_DATA_DIR,
            batch_size=BATCH_SIZE_STAGE2,
            sequence_length=SEQUENCE_LENGTH,
            input_shape=(INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS),
            shuffle=False # Pas besoin de mélanger pour la validation
        )
    
    if not train_seq_generator.sequences_data:
         print("Erreur: Aucune donnée d'entraînement n'a pu être chargée par le générateur. Vérifiez les fichiers vidéo et CSV.")
         return


    # 3. Créer le modèle combiné FOMO-TD-RNN
    # IMPORTANT: Assurez-vous que create_fomo_td_with_rnn_combined_model dans fomo_trainer.py
    # est modifié pour que la tête RNN ait NUM_MOTION_OUTPUTS (1) sortie avec activation 'tanh'.
    combined_model = create_fomo_td_with_rnn_combined_model(
        input_sequence_shape=COMBINED_MODEL_INPUT_SHAPE,
        fomo_num_classes=1, # NUM_OBJECT_CLASSES_FOMO (personne)
        motion_num_classes=NUM_MOTION_OUTPUTS, # Doit être 1 pour la régression
        fomo_alpha=FOMO_ALPHA,
        fomo_backbone_cutoff_layer_name=FOMO_BACKBONE_CUTOFF_LAYER_NAME,
        rnn_type=RNN_TYPE_STAGE2
    )
    print("Modèle combiné FOMO-TD-RNN (pour régression) créé.")
    combined_model.summary()

    # 4. Transférer les poids du modèle FOMO de phase 1 vers le sous-modèle FOMO dans TimeDistributed
    print("Tentative de transfert des poids du modèle FOMO de phase 1...")
    try:
        # Le sous-modèle FOMO (backbone + tête) est encapsulé dans la couche TimeDistributed
        time_distributed_fomo_layer = combined_model.get_layer('time_distributed_fomo_processing')
        fomo_processing_sub_model = time_distributed_fomo_layer.layer # C'est le fomo_processing_model

        # Assigner les poids du modèle FOMO de phase 1 directement au sous-modèle
        # Cela suppose que fomo_stage1_model et fomo_processing_sub_model sont architecturalement identiques.
        fomo_processing_sub_model.set_weights(fomo_stage1_model.get_weights())
        print("Poids du modèle FOMO de phase 1 transférés au sous-modèle FOMO du modèle combiné.")

    except Exception as e:
        print(f"Avertissement: Erreur lors du transfert des poids du modèle FOMO de phase 1: {e}")
        print("L'entraînement de la phase 2 commencera avec un sous-modèle FOMO initialisé aléatoirement (ou pré-entraîné ImageNet).")


    # 5. Geler les couches du sous-modèle FOMO (qui est dans TimeDistributed)
    print("Gel du sous-modèle FOMO (partie TimeDistributed)...")
    fomo_processing_sub_model_to_freeze = combined_model.get_layer('time_distributed_fomo_processing').layer
    fomo_processing_sub_model_to_freeze.trainable = False
    
    # La sortie 'fomo_output' du modèle combiné est une couche Lambda et n'a pas de poids à geler.
    # Les couches produisant les données pour cette sortie sont dans fomo_processing_sub_model_to_freeze.
    
    print("Sous-modèle FOMO gelé. Seule la tête RNN sera entraînée initialement.")
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
            'motion_output': tf.keras.losses.MeanSquaredError() # Perte pour la régression
        },
        # loss_weights={'fomo_output': 0.0, 'motion_output': 1.0}, # Explicitement si besoin
        metrics={'motion_output': [tf.keras.metrics.MeanAbsoluteError()]} # Métrique pour la régression
    )

    # 7. Entraîner le modèle (principalement la tête RNN)
    print("Début de l'entraînement de la phase 2 (tête RNN pour régression)...")
    
    monitor_metric = 'val_loss' if validation_seq_generator and validation_seq_generator.sequences_data else 'loss'
    if not (validation_seq_generator and validation_seq_generator.sequences_data):
        print("Attention: Pas de données de validation, ModelCheckpoint et EarlyStopping surveilleront 'loss'.")

    try:
        combined_model.fit(
            train_seq_generator,
            epochs=EPOCHS_STAGE2,
            validation_data=validation_seq_generator if validation_seq_generator and validation_seq_generator.sequences_data else None,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint(COMBINED_MODEL_SAVE_PATH, save_best_only=True, monitor=monitor_metric),
                tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_metric, factor=0.2, patience=5, min_lr=0.00001), # Augmenté patience
                tf.keras.callbacks.EarlyStopping(monitor=monitor_metric, patience=10, restore_best_weights=True) # Augmenté patience
            ]
        )
        print("Entraînement de la phase 2 terminé.")
        print(f"Modèle combiné sauvegardé dans {COMBINED_MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Erreur durant l'entraînement de la phase 2: {e}")
        # Lever l'exception pour un débogage plus facile si nécessaire
        # raise e 

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
