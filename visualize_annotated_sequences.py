import cv2
import numpy as np
import pandas as pd
import argparse
import os

# --- Configuration (inspirée de train_combined_model_stage2.py) ---
# Répertoire par défaut pour les fichiers CSV d'annotations.
# Peut être surchargé par une variable d'environnement ou un argument de script.
DEFAULT_ANNOTATION_DATA_DIR = '/Users/scampion/src/sport_video_scrapper/camera_movement_reports'
ANNOTATION_DATA_DIR = os.environ.get("ANNOTATION_DATA_DIR", DEFAULT_ANNOTATION_DATA_DIR)

SEQUENCE_LENGTH = 10 # Doit correspondre à la longueur de séquence utilisée lors de la génération des annotations

# --- Fonctions Utilitaires ---

def display_sequence_info(frame, frame_idx, sequence_length, move_x_value):
    """Affiche les informations de la séquence et l'annotation sur l'image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    text_color_info = (255, 255, 0) # Cyan pour info
    text_color_label = (0, 255, 0) # Vert pour label
    thickness = 2
    
    # Afficher l'index de l'image actuelle
    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), font, font_scale, text_color_info, thickness)

    if move_x_value is not None:
        # Cette image est la fin d'une séquence
        start_seq_frame_idx = frame_idx - sequence_length + 1
        display_text_sequence = f"Seq: [{start_seq_frame_idx}-{frame_idx}]"
        display_text_label = f"Move_X: {move_x_value:.3f}"
        
        cv2.putText(frame, display_text_sequence, (10, 60), font, font_scale, text_color_info, thickness)
        cv2.putText(frame, display_text_label, (10, 90), font, font_scale, text_color_label, thickness)
        
        # Dessiner une flèche indicative du mouvement (similaire à simulate_camera_movement.py)
        frame_height, frame_width = frame.shape[:2]
        arrow_color = (0, 0, 255) # Rouge
        arrow_center_y = 120
        arrow_base_length = frame_width // 5
        arrow_start_x = frame_width // 2
        arrow_end_x = arrow_start_x + int(move_x_value * arrow_base_length)
        arrow_end_x = np.clip(arrow_end_x, arrow_base_length // 4, frame_width - (arrow_base_length // 4))

        if abs(move_x_value) > 0.01: # Seuil pour dessiner la flèche
            cv2.arrowedLine(frame, (arrow_start_x, arrow_center_y), (arrow_end_x, arrow_center_y), 
                            arrow_color, thickness, tipLength=0.3)
        else:
            cv2.circle(frame, (arrow_start_x, arrow_center_y), 5, arrow_color, -1)
            
    return frame

# --- Logique Principale de Visualisation ---
def main(video_path, annotation_dir_override):
    global ANNOTATION_DATA_DIR # Permettre la modification de la variable globale
    if annotation_dir_override:
        ANNOTATION_DATA_DIR = annotation_dir_override
        print(f"Utilisation du répertoire d'annotations surchargé : {ANNOTATION_DATA_DIR}")
    else:
        print(f"Utilisation du répertoire d'annotations par défaut : {ANNOTATION_DATA_DIR}")

    # 1. Charger la Vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir le fichier vidéo : {video_path}")
        return

    total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 25 # Valeur par défaut si FPS non disponible
    print(f"Vidéo ouverte : {video_path}, Total Frames: {total_frames_video}, FPS: {fps:.2f}")

    # 2. Charger les Annotations
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
                # Ajuster la longueur des annotations si elle ne correspond pas au nombre de frames (comme dans le générateur)
                if len(move_x_annotations) > total_frames_video:
                    print(f"Attention: Plus d'annotations ({len(move_x_annotations)}) que de frames ({total_frames_video}). Troncation des annotations.")
                    move_x_annotations = move_x_annotations[:total_frames_video]
                elif len(move_x_annotations) < total_frames_video:
                     print(f"Attention: Moins d'annotations ({len(move_x_annotations)}) que de frames ({total_frames_video}). Certaines frames n'auront pas d'annotation.")
                     # Remplir avec NaN ou une valeur par défaut pour les frames manquantes pour éviter les erreurs d'index
                     move_x_annotations.extend([np.nan] * (total_frames_video - len(move_x_annotations)))


        except Exception as e:
            print(f"Erreur lors de la lecture du fichier CSV {annotation_path}: {e}. Affichage sans annotations de mouvement.")
            move_x_annotations = [] # Assurer que c'est une liste vide en cas d'erreur

    # 3. Boucle de Visualisation
    current_frame_idx = 0
    paused = False
    
    while True:
        if not paused or current_frame_idx == 0: # Lire une nouvelle frame si non en pause ou si c'est la première itération après une pause
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
            ret, frame = cap.read()
            if not ret:
                print(f"Fin de la vidéo atteinte ou erreur de lecture à la frame {current_frame_idx}.")
                current_frame_idx = max(0, current_frame_idx -1) # Revenir à la dernière frame valide
                paused = True # Forcer la pause à la fin
                # Re-lire la dernière frame valide pour l'affichage
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                ret, frame = cap.read()
                if not ret: # Si même la re-lecture échoue, sortir
                    break


        display_frame = frame.copy()
        
        # Récupérer l'annotation pour cette frame si elle est la fin d'une séquence
        current_move_x_value = None
        if current_frame_idx < len(move_x_annotations): # Vérifier si l'index est valide pour les annotations
            # Une annotation existe pour chaque frame. La séquence se termine à current_frame_idx.
            # La logique _extract_sequences_from_videos stocke (video_path, start_frame_idx_seq, motion_label)
            # où motion_label est celui de la dernière frame de la séquence.
            # Donc, move_x_annotations[current_frame_idx] est le label si current_frame_idx est la fin d'une séquence.
            if current_frame_idx >= SEQUENCE_LENGTH - 1:
                 # S'assurer que la valeur n'est pas NaN avant de l'utiliser
                if not pd.isna(move_x_annotations[current_frame_idx]):
                    current_move_x_value = move_x_annotations[current_frame_idx]
                else:
                    # Gérer le cas où l'annotation est NaN (par exemple, si le CSV avait des trous)
                    pass # current_move_x_value reste None

        display_frame = display_sequence_info(display_frame, current_frame_idx, SEQUENCE_LENGTH, current_move_x_value)
        cv2.imshow('Annotated Sequence Visualizer', display_frame)

        # Contrôles au clavier
        key = cv2.waitKey(int(1000/fps) if not paused else 0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '): # Espace pour Pause/Play
            paused = not paused
        elif key == ord('.'): # '.' pour frame suivante
            if current_frame_idx < total_frames_video - 1:
                current_frame_idx += 1
                paused = True # Mettre en pause lors du déplacement manuel
            else:
                print("Déjà à la dernière frame.")
        elif key == ord(','): # ',' pour frame précédente
            if current_frame_idx > 0:
                current_frame_idx -= 1
                paused = True # Mettre en pause lors du déplacement manuel
            else:
                print("Déjà à la première frame.")
        
        if not paused and current_frame_idx < total_frames_video -1 :
            current_frame_idx += 1
        elif not paused and current_frame_idx == total_frames_video -1:
            paused = True # Pause automatique à la dernière frame

    # 4. Nettoyage
    cap.release()
    cv2.destroyAllWindows()
    print("Visualisation terminée.")

# --- Point d'Entrée ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualise les séquences vidéo annotées utilisées pour l'entraînement du RNN.")
    parser.add_argument("video_path", help="Chemin vers le fichier vidéo d'entrée.")
    parser.add_argument("--annotation_dir", default=None,
                        help=f"Surcharger le répertoire des annotations CSV (défaut: {ANNOTATION_DATA_DIR}).")
    
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Erreur: Fichier vidéo non trouvé à {args.video_path}")
        exit(1)
    
    if args.annotation_dir and not os.path.isdir(args.annotation_dir):
        print(f"Erreur: Répertoire d'annotations spécifié non trouvé à {args.annotation_dir}")
        exit(1)

    main(args.video_path, args.annotation_dir)
