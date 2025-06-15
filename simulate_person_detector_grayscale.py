import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from fomo_trainer_grayscale import fomo_loss_function # Utiliser le trainer grayscale

# --- Configuration (doit correspondre aux paramètres d'entraînement du modèle grayscale) ---
MODEL_PATH = 'person_detector_fomo_grayscale.h5' # Modèle grayscale
VIDEO_DIR = 'data/test'
OUTPUT_VIDEO_DIR = 'data/test_simulated_grayscale' # Répertoire de sortie pour les simulations grayscale

INPUT_HEIGHT = 96
INPUT_WIDTH = 96
INPUT_CHANNELS = 1 # 1 canal pour grayscale
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)

SPATIAL_REDUCTION = 8     # Doit correspondre à la réduction utilisée par le modèle FOMO
GRID_HEIGHT = INPUT_HEIGHT // SPATIAL_REDUCTION
GRID_WIDTH = INPUT_WIDTH // SPATIAL_REDUCTION

PERSON_CLASS_ID = 0  # L'ID de classe pour 'personne' dans la sortie du modèle
NUM_CLASSES_MODEL_OUTPUT = 1 + 1 # Nombre de classes d'objets + 1 pour l'arrière-plan

DETECTION_THRESHOLD = 0.5  # Seuil de confiance pour considérer une détection (ajuster si nécessaire)
# --- Fin de la Configuration ---

def preprocess_frame(frame, target_shape):
    """Prétraite une image pour l'entrée du modèle (grayscale)."""
    # Convertir en niveaux de gris
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    img_resized = cv2.resize(gray_frame, (target_shape[1], target_shape[0]))
    img_normalized = img_resized.astype(np.float32) / 255.0
    # Ajouter une dimension de canal pour correspondre à l'entrée du modèle (H, W, 1)
    img_normalized = np.expand_dims(img_normalized, axis=-1)
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

    if heatmap.shape[-1] > 1: 
        heatmap_probs = tf.nn.softmax(heatmap, axis=-1).numpy()
    else: 
        heatmap_probs = tf.math.sigmoid(heatmap).numpy()

    person_heatmap = heatmap_probs[0, :, :, person_class_id]

    for r in range(grid_h):
        for c in range(grid_w):
            confidence = person_heatmap[r, c]
            if confidence > threshold:
                center_x_norm = (c + 0.5) / grid_w
                center_y_norm = (r + 0.5) / grid_h
                
                center_x_orig = int(center_x_norm * frame_w)
                center_y_orig = int(center_y_norm * frame_h)
                
                detected_centroids.append((center_x_orig, center_y_orig, confidence))
                
    return detected_centroids

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Erreur : Fichier modèle non trouvé à {MODEL_PATH}")
        return

    custom_objects = {'loss': fomo_loss_function(num_classes_with_background=NUM_CLASSES_MODEL_OUTPUT)}
    try:
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        print(f"Modèle chargé depuis {MODEL_PATH}")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
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

        original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        output_filename = os.path.join(OUTPUT_VIDEO_DIR, f"simulated_grayscale_{os.path.basename(video_path)}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out_video = cv2.VideoWriter(output_filename, fourcc, fps, (original_w, original_h))
        
        print(f"Sauvegarde de la vidéo simulée (grayscale) dans : {output_filename}")

        window_name = f"Detections FOMO Grayscale - {os.path.basename(video_path)}"
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Prétraitement de l'image pour le modèle grayscale
            processed_frame = preprocess_frame(frame.copy(), INPUT_SHAPE) # frame est BGR ici
            input_tensor = np.expand_dims(processed_frame, axis=0)

            heatmap_output = model.predict(input_tensor)
            
            centroids = postprocess_heatmap(
                heatmap_output,
                (original_h, original_w),
                (GRID_HEIGHT, GRID_WIDTH),
                PERSON_CLASS_ID,
                DETECTION_THRESHOLD
            )

            # Dessiner les détections sur l'image d'origine (couleur)
            display_frame = frame.copy()
            for (x, y, conf) in centroids:
                cv2.circle(display_frame, (x, y), 10, (0, 255, 0), 2) 
                cv2.putText(display_frame, f"{conf:.2f}", (x + 10, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Afficher une grille de 12 x 12 (optionnel, pour visualisation)
            grid_h_display = GRID_HEIGHT
            grid_w_display = GRID_WIDTH
            cell_height = original_h // grid_h_display
            cell_width = original_w // grid_w_display
            for r_idx in range(grid_h_display):
                for c_idx in range(grid_w_display):
                    cv2.rectangle(display_frame,
                                  (c_idx * cell_width, r_idx * cell_height),
                                  ((c_idx + 1) * cell_width, (r_idx + 1) * cell_height),
                                  (255, 0, 0), 1) # Grille en bleu

            cv2.imshow(window_name, display_frame)
            out_video.write(display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        out_video.release()
        cv2.destroyWindow(window_name)
        print(f"Vidéo simulée (grayscale) sauvegardée : {output_filename}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
             
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
