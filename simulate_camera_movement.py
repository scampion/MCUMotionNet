import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import argparse
import os

# --- Configuration (doit correspondre aux paramètres d'entraînement) ---
DEFAULT_MODEL_PATH = 'fomo_td_rnn_regression_stage2.h5'
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

def display_predicted_movement(frame, movement_x_value):
    """Affiche la valeur du mouvement X prédit et une flèche sur l'image."""
    display_text = f"Pred Move X: {movement_x_value:.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_color = (0, 255, 0) # Vert
    arrow_color = (0, 0, 255) # Rouge
    thickness = 2
    
    # Position du texte (en haut à gauche)
    cv2.putText(frame, display_text, (10, 30), font, font_scale, text_color, thickness)

    # Dessiner une flèche indicative du mouvement
    frame_height, frame_width = frame.shape[:2]
    arrow_center_y = 60
    arrow_base_length = frame_width // 4 # Longueur de la flèche si mouvement max
    
    # Point de départ de la flèche (au centre horizontalement, sous le texte)
    arrow_start_x = frame_width // 2
    
    # Calculer le point d'arrivée de la flèche basé sur movement_x_value
    # movement_x_value est entre -1 (gauche) et 1 (droite)
    arrow_end_x = arrow_start_x + int(movement_x_value * arrow_base_length)
    
    # S'assurer que la flèche ne sort pas trop de l'écran (optionnel, pour esthétique)
    arrow_end_x = np.clip(arrow_end_x, arrow_base_length // 4, frame_width - (arrow_base_length // 4))

    if abs(movement_x_value) > 0.05: # Ne dessiner la flèche que si le mouvement est significatif
        cv2.arrowedLine(frame, (arrow_start_x, arrow_center_y), (arrow_end_x, arrow_center_y), 
                        arrow_color, thickness, tipLength=0.3)
    else: # Dessiner un point si pas de mouvement significatif
        cv2.circle(frame, (arrow_start_x, arrow_center_y), 5, arrow_color, -1)
        
    return frame

# --- Logique Principale de Simulation ---
def main(video_path, model_path):
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
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Vidéo ouverte : {frame_width}x{frame_height} @ {fps:.2f} FPS")

    # File d'attente pour stocker la séquence d'images prétraitées
    frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
    
    # Initialiser la file d'attente avec des images noires si besoin pour les premières prédictions
    # ou attendre que la file soit pleine. Ici, nous attendrons.
    # black_frame_processed = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS), dtype=np.float32)
    # for _ in range(SEQUENCE_LENGTH):
    #     frame_sequence.append(black_frame_processed)

    predicted_movement_x = 0.0 # Valeur initiale

    # 3. Boucle de Traitement des Images
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fin de la vidéo ou erreur de lecture.")
            break

        original_frame_for_display = frame.copy()

        # Prétraiter l'image
        processed_frame = preprocess_single_frame(frame, (INPUT_HEIGHT, INPUT_WIDTH))
        frame_sequence.append(processed_frame)

        # Si la séquence est complète, faire une prédiction
        if len(frame_sequence) == SEQUENCE_LENGTH:
            # Convertir la séquence en un batch de taille 1 pour le modèle
            input_tensor = np.expand_dims(np.array(list(frame_sequence)), axis=0)
            
            try:
                predictions = model.predict(input_tensor, verbose=0) # verbose=0 pour moins de logs Keras
                # La sortie 'motion_output' est un tenseur (batch_size, 1)
                predicted_movement_x = predictions['motion_output'][0][0] 
            except Exception as e:
                print(f"Erreur durant la prédiction : {e}")
                # Continuer avec la dernière valeur prédite ou une valeur neutre
                # predicted_movement_x = 0.0 
        
        # Afficher l'image avec la prédiction
        display_frame = display_predicted_movement(original_frame_for_display, predicted_movement_x)
        cv2.imshow('Camera Movement Simulation', display_frame)

        # Quitter avec la touche 'q'
        if cv2.waitKey(int(1000/fps) if fps > 0 else 25) & 0xFF == ord('q'):
            break

    # 4. Nettoyage
    cap.release()
    cv2.destroyAllWindows()
    print("Simulation terminée.")

# --- Point d'Entrée ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simule la prédiction de mouvement de caméra sur une vidéo.")
    parser.add_argument("video_path", help="Chemin vers le fichier vidéo d'entrée.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH, 
                        help=f"Chemin vers le fichier de modèle .h5 entraîné (défaut: {DEFAULT_MODEL_PATH}).")
    
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

    main(args.video_path, args.model_path)
