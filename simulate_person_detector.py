import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from fomo_trainer import fomo_loss_function # Nécessaire pour charger le modèle avec une perte personnalisée

# --- Configuration (doit correspondre aux paramètres d'entraînement) ---
MODEL_PATH = 'person_detector_fomo.h5'
VIDEO_DIR = 'data/test'
OUTPUT_VIDEO_DIR = 'data/test_simulated' # Répertoire pour sauvegarder les vidéos simulées

INPUT_HEIGHT = 96
INPUT_WIDTH = 96
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, 3) # 3 canaux pour RGB

SPATIAL_REDUCTION = 8     # Doit correspondre à la réduction utilisée par le modèle FOMO
GRID_HEIGHT = INPUT_HEIGHT // SPATIAL_REDUCTION
GRID_WIDTH = INPUT_WIDTH // SPATIAL_REDUCTION

PERSON_CLASS_ID = 0  # L'ID de classe pour 'personne' dans la sortie du modèle
NUM_CLASSES_MODEL_OUTPUT = 1 + 1 # Nombre de classes d'objets + 1 pour l'arrière-plan

DETECTION_THRESHOLD = 0.8  # Seuil de confiance pour considérer une détection
# --- Fin de la Configuration ---

def preprocess_frame(frame, target_shape):
    """Prétraite une image pour l'entrée du modèle."""
    img_resized = cv2.resize(frame, (target_shape[1], target_shape[0]))
    img_normalized = img_resized.astype(np.float32) / 255.0
    return img_normalized

def postprocess_heatmap(heatmap, original_frame_shape, grid_shape, person_class_id, threshold):
    """
    Post-traite la carte de chaleur pour trouver les centroïdes.
    heatmap: Sortie brute du modèle (logits ou probabilités).
    original_frame_shape: Tuple (height, width) de l'image d'origine.
    grid_shape: Tuple (grid_h, grid_w) de la carte de chaleur.
    person_class_id: ID de la classe 'personne'.
    threshold: Seuil de confiance.
    """
    detected_centroids = []
    grid_h, grid_w = grid_shape
    frame_h, frame_w = original_frame_shape

    # Si la sortie du modèle sont des logits, appliquer softmax
    # La sortie de notre modèle FOMO est déjà des logits, la fonction de perte s'attend à cela.
    # Pour l'inférence, nous avons besoin de probabilités.
    if heatmap.shape[-1] > 1: # Vérifier si plusieurs classes (y compris l'arrière-plan)
        heatmap_probs = tf.nn.softmax(heatmap, axis=-1).numpy()
    else: # Cas d'une seule classe en sortie (peu probable avec FOMO et arrière-plan)
        heatmap_probs = tf.math.sigmoid(heatmap).numpy()


    # La sortie de predict() a une dimension de batch, donc heatmap_probs[0]
    person_heatmap = heatmap_probs[0, :, :, person_class_id]

    for r in range(grid_h):
        for c in range(grid_w):
            confidence = person_heatmap[r, c]
            if confidence > threshold:
                # Calculer les coordonnées du centroïde dans l'image d'origine
                center_x_norm = (c + 0.5) / grid_w  # +0.5 pour le centre de la cellule
                center_y_norm = (r + 0.5) / grid_h
                
                center_x_orig = int(center_x_norm * frame_w)
                center_y_orig = int(center_y_norm * frame_h)
                
                detected_centroids.append((center_x_orig, center_y_orig, confidence))
                
    return detected_centroids

def main():
    # Charger le modèle
    if not os.path.exists(MODEL_PATH):
        print(f"Erreur : Fichier modèle non trouvé à {MODEL_PATH}")
        return

    # La fonction de perte est nécessaire si elle a été utilisée pendant l'entraînement et la sauvegarde
    # et n'est pas une perte Keras standard.
    custom_objects = {'loss': fomo_loss_function(num_classes_with_background=NUM_CLASSES_MODEL_OUTPUT)}
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        print(f"Modèle chargé depuis {MODEL_PATH}")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        print("Assurez-vous que la fonction de perte personnalisée est correctement définie si nécessaire.")
        return

    model.summary()

    if not os.path.exists(OUTPUT_VIDEO_DIR):
        os.makedirs(OUTPUT_VIDEO_DIR)
        print(f"Répertoire de sortie créé : {OUTPUT_VIDEO_DIR}")

    video_files = glob.glob(os.path.join(VIDEO_DIR, '*.mp4'))
    if not video_files:
        print(f"Aucune vidéo MP4 trouvée dans {VIDEO_DIR}")
        return

    for video_path in video_files:
        print(f"Traitement de la vidéo : {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Erreur : Impossible d'ouvrir la vidéo {video_path}")
            continue

        # Configuration pour l'écriture de la vidéo de sortie
        original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        output_filename = os.path.join(OUTPUT_VIDEO_DIR, f"simulated_{os.path.basename(video_path)}")
        # Utiliser 'mp4v' comme codec pour les fichiers .mp4. D'autres options sont 'XVID' pour .avi
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out_video = cv2.VideoWriter(output_filename, fourcc, fps, (original_w, original_h))
        
        print(f"Sauvegarde de la vidéo simulée dans : {output_filename}")

        window_name = f"Detections FOMO - {os.path.basename(video_path)}"
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            original_h, original_w = frame.shape[:2]
            
            # Prétraitement de l'image
            processed_frame = preprocess_frame(frame.copy(), INPUT_SHAPE)
            input_tensor = np.expand_dims(processed_frame, axis=0)

            # Inférence
            heatmap_output = model.predict(input_tensor)
            
            # Post-traitement
            centroids = postprocess_heatmap(
                heatmap_output,
                (original_h, original_w),
                (GRID_HEIGHT, GRID_WIDTH),
                PERSON_CLASS_ID,
                DETECTION_THRESHOLD
            )

            # Dessiner les détections sur l'image
            display_frame = frame.copy()
            for (x, y, conf) in centroids:
                cv2.circle(display_frame, (x, y), 10, (0, 255, 0), 2) # Cercle vert
                cv2.putText(display_frame, f"{conf:.2f}", (x + 10, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Afficher une grille de 12 x 12
            grid_h = GRID_HEIGHT
            grid_w = GRID_WIDTH
            cell_height = original_h // grid_h
            cell_width = original_w // grid_w
            for r in range(grid_h):
                for c in range(grid_w):
                    cv2.rectangle(display_frame,
                                  (c * cell_width, r * cell_height),
                                  ((c + 1) * cell_width, (r + 1) * cell_height),
                                  (255, 0, 0), 1)

            # Afficher l'image
            cv2.imshow(window_name, display_frame)
            
            # Écrire l'image dans le fichier vidéo de sortie
            out_video.write(display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): # Appuyer sur 'q' pour quitter
                break
        
        cap.release()
        out_video.release() # Libérer l'objet VideoWriter
        cv2.destroyWindow(window_name) # Fermer la fenêtre spécifique à la vidéo
        print(f"Vidéo simulée sauvegardée : {output_filename}")
        if cv2.waitKey(1) & 0xFF == ord('q'): # Permettre de quitter entre les vidéos
             break
             
    cv2.destroyAllWindows() # S'assurer que toutes les fenêtres sont fermées à la fin

if __name__ == '__main__':
    main()
