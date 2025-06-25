import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import argparse
import os
import pandas as pd # Ajout de pandas

# --- Configuration (doit correspondre aux paramètres d'entraînement) ---
DEFAULT_MODEL_PATH = 'fomo_td_rnn_regression_stage2.h5'
# Répertoire par défaut pour les fichiers CSV d'annotations.
DEFAULT_ANNOTATION_DATA_DIR = '/Users/scampion/src/sport_video_scrapper/camera_movement_reports' # Ajustez si nécessaire
ANNOTATION_DATA_DIR = os.environ.get("ANNOTATION_DATA_DIR", DEFAULT_ANNOTATION_DATA_DIR)
INPUT_HEIGHT = 96
INPUT_WIDTH = 96
INPUT_CHANNELS = 3
SEQUENCE_LENGTH = 10 # Doit correspondre à la longueur de séquence utilisée pour l'entraînement

# --- Fonctions Utilitaires ---
def preprocess_single_frame(frame, target_shape_hw):
    """Prétraite une seule image pour l'entrée du modèle."""
    # target_shape_hw est (height, width)
    img_resized = cv2.resize(frame, (target_shape_hw[1], target_shape_hw[0]))
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized

def display_movement_info(frame, predicted_movement_x, annotated_movement_x):
    """Affiche les valeurs de mouvement X prédit et annoté, ainsi que des flèches sur l'image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # --- Prédiction ---
    pred_text = f"Pred Move X: {predicted_movement_x:.3f}"
    pred_text_color = (0, 255, 0) # Vert
    pred_arrow_color = (0, 0, 255) # Rouge
    cv2.putText(frame, pred_text, (10, 30), font, font_scale, pred_text_color, thickness)

    frame_height, frame_width = frame.shape[:2]
    arrow_center_y_pred = 60
    arrow_base_length = frame_width // 5 # Longueur de base pour les flèches
    
    arrow_start_x = frame_width // 2 
    
    # Flèche pour la prédiction
    pred_arrow_end_x = arrow_start_x + int(predicted_movement_x * arrow_base_length)
    pred_arrow_end_x = np.clip(pred_arrow_end_x, arrow_base_length // 4, frame_width - (arrow_base_length // 4))
    if abs(predicted_movement_x) > 0.05:
        cv2.arrowedLine(frame, (arrow_start_x, arrow_center_y_pred), (pred_arrow_end_x, arrow_center_y_pred), 
                        pred_arrow_color, thickness, tipLength=0.3)
    else:
        cv2.circle(frame, (arrow_start_x, arrow_center_y_pred), 5, pred_arrow_color, -1)

    # --- Annotation ---
    if annotated_movement_x is not None:
        annot_text = f"Annot Move X: {annotated_movement_x:.3f}"
        annot_text_color = (255, 165, 0) # Orange-ish Bleu clair (pour contraste)
        annot_arrow_color = (255, 0, 0)   # Bleu
        cv2.putText(frame, annot_text, (10, 90), font, font_scale, annot_text_color, thickness)
        
        arrow_center_y_annot = 120
        # Flèche pour l'annotation
        annot_arrow_end_x = arrow_start_x + int(annotated_movement_x * arrow_base_length)
        annot_arrow_end_x = np.clip(annot_arrow_end_x, arrow_base_length // 4, frame_width - (arrow_base_length // 4))
        if abs(annotated_movement_x) > 0.01: # Seuil différent ou identique, au choix
            cv2.arrowedLine(frame, (arrow_start_x, arrow_center_y_annot), (annot_arrow_end_x, arrow_center_y_annot), 
                            annot_arrow_color, thickness, tipLength=0.3)
        else:
            cv2.circle(frame, (arrow_start_x, arrow_center_y_annot), 5, annot_arrow_color, -1)
            
    return frame

# --- Logique Principale de Simulation ---
def main(video_path, model_path, annotation_dir_override, output_video_path=None):
    global ANNOTATION_DATA_DIR
    if annotation_dir_override:
        ANNOTATION_DATA_DIR = annotation_dir_override
        print(f"Utilisation du répertoire d'annotations surchargé : {ANNOTATION_DATA_DIR}")
    else:
        print(f"Utilisation du répertoire d'annotations par défaut : {ANNOTATION_DATA_DIR}")
    # 1. Charger le Modèle
    print(f"Chargement du modèle depuis : {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
        print("Modèle chargé avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return

    # 2. Initialiser la Capture Vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir le fichier vidéo : {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 25 # Default FPS if not available
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Vidéo ouverte : {frame_width}x{frame_height} @ {fps:.2f} FPS, Total Frames: {total_frames_video}")

    # Initialiser VideoWriter si un chemin de sortie est fourni
    video_writer = None
    if output_video_path:
        # Utiliser un codec commun comme MP4V pour .mp4 ou XVID pour .avi
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # ou 'XVID'
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        if not video_writer.isOpened():
            print(f"Erreur: Impossible d'ouvrir VideoWriter pour le chemin: {output_video_path}")
            # Optionnel: décider si l'on doit quitter ou juste continuer sans sauvegarder
            video_writer = None # S'assurer qu'il est None pour ne pas tenter d'écrire
        else:
            print(f"Sauvegarde de la vidéo visualisée dans : {output_video_path}")

    # 3. Charger les Annotations
    video_filename = os.path.basename(video_path)
    video_name_without_ext = os.path.splitext(video_filename)[0]
    annotation_path = os.path.join(ANNOTATION_DATA_DIR, f"{video_name_without_ext}_movement.csv")
    
    move_x_annotations = []
    if not os.path.exists(annotation_path):
        print(f"Attention: Fichier d'annotation CSV non trouvé à {annotation_path}. Affichage sans annotations de mouvement.")
    else:
        try:
            annotations_df = pd.read_csv(annotation_path)
            if 'Move_X' not in annotations_df.columns:
                print(f"Attention: Colonne 'Move_X' non trouvée dans {annotation_path}. Affichage sans annotations de mouvement.")
            else:
                move_x_annotations = annotations_df['Move_X'].tolist()
                print(f"Annotations chargées depuis {annotation_path}. Nombre d'annotations: {len(move_x_annotations)}")
                # Ajuster la longueur des annotations si nécessaire (comme dans visualize_annotated_sequences)
                if len(move_x_annotations) > total_frames_video:
                    move_x_annotations = move_x_annotations[:total_frames_video]
                elif len(move_x_annotations) < total_frames_video:
                    move_x_annotations.extend([np.nan] * (total_frames_video - len(move_x_annotations)))
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier CSV {annotation_path}: {e}. Affichage sans annotations de mouvement.")

    # File d'attente pour stocker la séquence d'images prétraitées
    frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
    predicted_movement_x = 0.0 # Valeur initiale pour la prédiction
    current_frame_idx = 0

    # 4. Boucle de Traitement des Images
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin de la vidéo ou erreur de lecture.")
            break

        original_frame_for_display = frame.copy()

        # Prétraiter l'image
        processed_frame = preprocess_single_frame(frame, (INPUT_HEIGHT, INPUT_WIDTH))
        frame_sequence.append(processed_frame)

        current_annotated_movement_x = None
        # Si la séquence est complète, faire une prédiction
        if len(frame_sequence) == SEQUENCE_LENGTH:
            input_tensor = np.expand_dims(np.array(list(frame_sequence)), axis=0)
            try:
                predictions = model.predict(input_tensor, verbose=0)
                predicted_movement_x = predictions['motion_output'][0][0]
            except Exception as e:
                print(f"Erreur durant la prédiction : {e}")
        
        # Récupérer l'annotation pour la frame actuelle (qui est la fin de la séquence)
        if current_frame_idx < len(move_x_annotations):
            annot_val = move_x_annotations[current_frame_idx]
            if not pd.isna(annot_val):
                current_annotated_movement_x = annot_val
        
        # Afficher l'image avec la prédiction et l'annotation
        display_frame = display_movement_info(original_frame_for_display, predicted_movement_x, current_annotated_movement_x)
        cv2.imshow('Camera Movement Simulation & Annotation', display_frame)

        # Sauvegarder l'image si VideoWriter est initialisé
        if video_writer:
            video_writer.write(display_frame)
        
        current_frame_idx += 1
        # Quitter avec la touche 'q'
        if cv2.waitKey(int(1000/fps) if fps > 0 else 25) & 0xFF == ord('q'):
            break

    # 4. Nettoyage
    cap.release()
    if video_writer:
        video_writer.release()
        print(f"Vidéo de sortie sauvegardée dans {output_video_path}")
    cv2.destroyAllWindows()
    print("Simulation terminée.")

# --- Point d'Entrée ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simule la prédiction de mouvement de caméra sur une vidéo et affiche les annotations.")
    parser.add_argument("video_path", help="Chemin vers le fichier vidéo d'entrée.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, 
                        help=f"Chemin vers le fichier de modèle .h5 entraîné (défaut: {DEFAULT_MODEL_PATH}).")
    parser.add_argument("--annotation_dir", default=None,
                        help=f"Surcharger le répertoire des annotations CSV (défaut: {DEFAULT_ANNOTATION_DATA_DIR}).")
    parser.add_argument("--output_video", default=None,
                        help="Chemin optionnel pour sauvegarder la vidéo visualisée (ex: output.mp4).")
    
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Erreur: Fichier vidéo non trouvé à {args.video_path}")
        exit(1)
    if not os.path.exists(args.model_path):
        print(f"Erreur: Fichier modèle non trouvé à {args.model_path}")
        # Essayer de chercher dans le répertoire du script si un chemin absolu n'est pas donné
        script_dir_model_path = os.path.join(os.path.dirname(__file__), args.model_path)
        if os.path.exists(script_dir_model_path):
            args.model_path = script_dir_model_path
        else:
            print(f"  (également non trouvé à {script_dir_model_path})")
            exit(1)
    
    if args.annotation_dir and not os.path.isdir(args.annotation_dir):
        print(f"Erreur: Répertoire d'annotations spécifié non trouvé à {args.annotation_dir}")
        exit(1)

    main(args.video_path, args.model_path, args.annotation_dir, args.output_video)
