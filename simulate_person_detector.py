import os
import glob
import cv2
import numpy as np
import tensorflow as tf
import math # Ajouté pour sqrt
from fomo_trainer import fomo_loss_function # Nécessaire pour charger le modèle avec une perte personnalisée

# --- Configuration (doit correspondre aux paramètres d'entraînement) ---
# MODEL_PATH = 'person_detector_fomo.h5' # Ancien chemin, maintenant remplacé par COMBINED_MODEL_PATH
VIDEO_DIR = 'data/test'
OUTPUT_VIDEO_DIR = 'data/test_simulated' # Répertoire pour sauvegarder les vidéos simulées

INPUT_HEIGHT = 96
INPUT_WIDTH = 96
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, 3) # 3 canaux pour RGB

SPATIAL_REDUCTION = 4     # Doit correspondre à la réduction utilisée par le modèle FOMO
GRID_HEIGHT = INPUT_HEIGHT // SPATIAL_REDUCTION
GRID_WIDTH = INPUT_WIDTH // SPATIAL_REDUCTION

PERSON_CLASS_ID = 0  # L'ID de classe pour 'personne' dans la sortie du modèle
NUM_CLASSES_MODEL_OUTPUT = 1 + 1 # Nombre de classes d'objets + 1 pour l'arrière-plan

DETECTION_THRESHOLD = 0.8  # Seuil de confiance pour considérer une détection

# Configuration pour le suivi des centroïdes et la prédiction de panoramique
CENTROID_MATCHING_THRESHOLD_DISTANCE = INPUT_WIDTH / 4  # Distance max pour apparier les centroïdes entre les images
HIST_NUM_BINS = 10  # Nombre de classes pour l'histogramme de mouvement horizontal
# Plage pour l'histogramme: de -moitié_largeur_input à +moitié_largeur_input
HIST_RANGE = (-INPUT_WIDTH // 2, INPUT_WIDTH // 2)
# Seuil pour la différence de mouvement dans l'histogramme pour prédire un panoramique
PAN_PREDICTION_HIST_DIFF_THRESHOLD = HIST_NUM_BINS // 5 # Exemple: si 20% de plus de mouvement dans une direction
MOTION_HIST_ZERO_THRESHOLD = INPUT_WIDTH / 20 # Marge pour considérer un mouvement comme non nul

# Configuration pour le modèle combiné FOMO-RNN
COMBINED_MODEL_PATH = 'fomo_rnn_combined_model.h5' # Chemin vers le modèle combiné entraîné
SEQUENCE_LENGTH = 10  # Nombre d'images consécutives pour la prédiction de mouvement
# Les classes de mouvement attendues par la tête RNN du modèle combiné
MOTION_CLASSES = ["Tourner Gauche", "Rester Statique", "Tourner Droite"] 
# Assurez-vous que NUM_CLASSES_MODEL_OUTPUT est correct pour la tête FOMO du modèle combiné
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
                
                # Retourner aussi les coordonnées de la grille (r, c)
                detected_centroids.append((center_x_orig, center_y_orig, confidence, r, c))
                
    return detected_centroids

def compute_optical_flow(prev_centroids_data_with_grid, current_centroids_data_with_grid, grid_shape):
    """
    Calcule le flux optique basé sur la disparition/apparition de centroïdes dans les cellules de la grille.
    Retourne les paires de points de flux et les déplacements horizontaux.
    prev_centroids_data_with_grid: liste de (x, y, conf, r_grid, c_grid) de l'image N-1
    current_centroids_data_with_grid: liste de (x, y, conf, r_grid, c_grid) de l'image N
    grid_shape: (GRID_HEIGHT, GRID_WIDTH)
    """
    horizontal_displacements = []
    matched_pairs = []
    grid_h, grid_w = grid_shape

    prev_grid_map = {(r, c): (x, y, conf) for x, y, conf, r, c in prev_centroids_data_with_grid}
    current_grid_map = {(r, c): (x, y, conf) for x, y, conf, r, c in current_centroids_data_with_grid}

    prev_occupied_cells = set(prev_grid_map.keys())
    current_occupied_cells = set(current_grid_map.keys())

    disappeared_cells = prev_occupied_cells - current_occupied_cells
    appeared_cells = current_occupied_cells - prev_occupied_cells

    for prev_r, prev_c in disappeared_cells:
        prev_x, prev_y, _ = prev_grid_map[(prev_r, prev_c)]
        
        neighboring_appeared_centroids_coords = []
        # Définir les 8 voisins (incluant diagonales)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue # Cellule elle-même
                
                neighbor_r, neighbor_c = prev_r + dr, prev_c + dc
                
                if 0 <= neighbor_r < grid_h and 0 <= neighbor_c < grid_w:
                    if (neighbor_r, neighbor_c) in appeared_cells:
                        # Ce voisin est une cellule nouvellement apparue
                        new_x, new_y, _ = current_grid_map[(neighbor_r, neighbor_c)]
                        neighboring_appeared_centroids_coords.append((new_x, new_y))
        
        if neighboring_appeared_centroids_coords:
            # Calculer le centroïde moyen des voisins apparus
            avg_new_x = sum(c[0] for c in neighboring_appeared_centroids_coords) / len(neighboring_appeared_centroids_coords)
            avg_new_y = sum(c[1] for c in neighboring_appeared_centroids_coords) / len(neighboring_appeared_centroids_coords)
            
            dx = avg_new_x - prev_x
            horizontal_displacements.append(dx)
            matched_pairs.append(((prev_x, prev_y), (avg_new_x, avg_new_y)))
            
    return matched_pairs, horizontal_displacements

def draw_optical_flow_arrows(display_frame, matched_pairs):
    """
    Dessine des flèches de flux optique sur l'image pour le débogage.
    Prend en entrée l'image et les paires de centroïdes appariés.
    """
    for pair in matched_pairs:
        (prev_x, prev_y), (curr_x, curr_y) = pair
        cv2.arrowedLine(display_frame, 
                        (int(prev_x), int(prev_y)), 
                        (int(curr_x), int(curr_y)), 
                        (0, 0, 255), 2, tipLength=0.3)
    return display_frame

def main():
    # Charger le modèle combiné FOMO-RNN
    if not os.path.exists(COMBINED_MODEL_PATH):
        print(f"Erreur : Fichier modèle combiné non trouvé à {COMBINED_MODEL_PATH}")
        return

    # Pour charger le modèle combiné, vous pourriez avoir besoin de spécifier des custom_objects
    # si des fonctions de perte personnalisées ont été utilisées pour chaque sortie pendant l'entraînement.
    # Pour l'instant, supposons que la fonction de perte FOMO est la seule pertinente pour le chargement
    # si le modèle a été sauvegardé après compilation avec des noms de pertes standards pour la partie mouvement.
    # Si le modèle a été sauvegardé avec des noms pour ses sorties, la perte fomo_loss_function
    # doit être associée à la sortie fomo lors de la compilation du modèle combiné.
    # Exemple: losses = {'fomo_output': fomo_loss_function(...), 'motion_output': 'categorical_crossentropy'}
    # Pour le chargement, Keras essaie de reconstruire les pertes si elles ne sont pas standard.
    # Il est plus sûr de fournir les objets personnalisés si nécessaire.
    custom_objects_combined = {}
    # Si la sortie FOMO du modèle combiné s'appelle 'fomo_output' et utilise fomo_loss_function:
    # custom_objects_combined['fomo_output_loss'] = fomo_loss_function(num_classes_with_background=NUM_CLASSES_MODEL_OUTPUT) 
    # Ou simplement la fonction de perte si elle est référencée par son nom dans le modèle sauvegardé.
    custom_objects_combined['loss'] = fomo_loss_function(num_classes_with_background=NUM_CLASSES_MODEL_OUTPUT)


    combined_model = None
    try:
        # Si le modèle combiné a été sauvegardé sans sa compilation (juste les poids),
        # vous devriez recréer l'architecture ici en appelant create_fomo_rnn_combined_model
        # puis charger les poids avec combined_model.load_weights(COMBINED_MODEL_PATH).
        # Si sauvegardé comme un modèle complet (model.save()), load_model devrait fonctionner.
        combined_model = tf.keras.models.load_model(COMBINED_MODEL_PATH, custom_objects=custom_objects_combined, compile=False)
        # compile=False est souvent plus sûr si vous n'allez pas entraîner davantage ici.
        print(f"Modèle combiné FOMO-RNN chargé depuis {COMBINED_MODEL_PATH}")
        combined_model.summary()
    except Exception as e:
        print(f"Erreur lors du chargement du modèle combiné FOMO-RNN : {e}")
        print("Assurez-vous que le chemin est correct et que les custom_objects nécessaires sont fournis.")
        return

    prev_centroids_data = [] 
    camera_pan_prediction = "Initialisation..."
    frame_sequence = [] # Pour stocker la séquence d'images pour le modèle combiné

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
            processed_frame = preprocess_frame(frame.copy(), (INPUT_HEIGHT, INPUT_WIDTH, INPUT_SHAPE[2])) # Utiliser INPUT_SHAPE pour les dimensions

            # Gérer la séquence d'images pour le modèle combiné
            frame_sequence.append(processed_frame)
            if len(frame_sequence) > SEQUENCE_LENGTH:
                frame_sequence.pop(0) # Garder la séquence à la bonne longueur

            heatmap_output_fomo = None
            motion_prediction_probs = None

            if len(frame_sequence) == SEQUENCE_LENGTH and combined_model:
                # Préparer l'entrée pour le modèle combiné
                model_input_sequence = np.expand_dims(np.array(frame_sequence), axis=0) # (1, seq_length, H, W, C)
                
                try:
                    # Le modèle combiné retourne deux sorties sous forme de dictionnaire
                    predictions = combined_model.predict(model_input_sequence)
                    heatmap_output_fomo = predictions['fomo_output'] # Accès par clé
                    motion_prediction_probs = predictions['motion_output'] # Accès par clé

                    predicted_class_idx = np.argmax(motion_prediction_probs[0])
                    camera_pan_prediction = f"RNN: {MOTION_CLASSES[predicted_class_idx]}"

                except Exception as e:
                    print(f"Erreur de prédiction avec le modèle combiné: {e}")
                    camera_pan_prediction = "Combiné: Erreur"
                    # heatmap_output_fomo restera None, donc pas de détections FOMO affichées
            else:
                camera_pan_prediction = f"Combiné: Collecte ({len(frame_sequence)}/{SEQUENCE_LENGTH})"
                # Pas assez d'images dans la séquence pour prédire, ou modèle non chargé
                # heatmap_output_fomo restera None, donc pas de détections FOMO affichées dans ce cas.
                # Si vous voulez quand même une sortie FOMO pour chaque image même pendant la collecte :
                # vous auriez besoin d'un modèle FOMO séparé ou d'une logique pour passer une seule image.
                # Pour l'instant, la détection FOMO n'est active que lorsque la séquence est pleine.


            centroids = []
            if heatmap_output_fomo is not None:
                centroids = postprocess_heatmap(
                    heatmap_output_fomo, # Utiliser la sortie FOMO du modèle combiné
                    (original_h, original_w),
                    (GRID_HEIGHT, GRID_WIDTH),
                    PERSON_CLASS_ID,
                    DETECTION_THRESHOLD
                )

            display_frame = frame.copy()
            current_centroids_data = centroids 

            # Calculer le flux optique (pour affichage des flèches, si souhaité)
            # current_centroids_data contient maintenant (x, y, conf, r_grid, c_grid)
            # prev_centroids_data doit aussi suivre ce format
            matched_centroid_pairs, horizontal_displacements = compute_optical_flow(
                prev_centroids_data, # Doit être une liste de (x,y,conf,r,c)
                current_centroids_data, # Est déjà une liste de (x,y,conf,r,c)
                (GRID_HEIGHT, GRID_WIDTH)
            )
            # print horizontal_displacements # Pour débogage, afficher les déplacements horizontaux
            print("Horizontal Displacements:", horizontal_displacements)


            # Dessiner les flèches de flux optique (pour le débogage)
            # Vous pouvez commenter/décommenter cette ligne pour activer/désactiver l'affichage des flèches
            display_frame = draw_optical_flow_arrows(display_frame, matched_centroid_pairs)

            # La prédiction du mouvement de la caméra est maintenant gérée par le modèle combiné ci-dessus.
            # La logique de l'histogramme pour la prédiction n'est plus la source principale si combined_model est utilisé.
            # Nous calculons current_hist uniquement pour l'affichage.
            current_hist = np.zeros(HIST_NUM_BINS, dtype=np.float32) 
            if horizontal_displacements:
                hist_values, _ = np.histogram(horizontal_displacements, bins=HIST_NUM_BINS, range=HIST_RANGE)
                current_hist = hist_values.astype(np.float32)

            # Si combined_model n'est pas chargé ou si la séquence n'est pas pleine, 
            # camera_pan_prediction aura une valeur comme "Collecte..." ou "Initialisation...".
            # Si vous souhaitez un fallback à l'ancienne méthode ici, il faudrait ajouter une condition
            # explicite (par exemple, if not combined_model or len(frame_sequence) < SEQUENCE_LENGTH: ...)
            # Pour l'instant, la prédiction est soit "RNN: <mouvement>", "RNN: Collecte", "RNN: Erreur", ou "Initialisation...".

            prev_centroids_data = list(current_centroids_data) 

            # centroids est une liste de (x, y, conf, r, c)
            for (x, y, conf, r_grid, c_grid) in centroids: # Dépaqueter r_grid, c_grid même si non utilisés ici
                cv2.circle(display_frame, (x, y), 10, (0, 255, 0), 2) # Cercle vert
                cv2.putText(display_frame, f"{conf:.2f}", (x + 10, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Afficher la prédiction du mouvement panoramique
            cv2.putText(display_frame, camera_pan_prediction, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Préparer la zone d'affichage de l'histogramme des mouvements horizontaux
            # current_hist contient l'histogramme du pas de temps actuel
            hist_display_height = 50
            hist_display_width = INPUT_WIDTH # ou une autre largeur appropriée
            hist_img = np.zeros((hist_display_height, hist_display_width, 3), dtype=np.uint8) # Fond noir par défaut

            if np.any(current_hist): # Si current_hist n'est pas que des zéros (signifie qu'il y a eu des déplacements)
                # Normaliser l'histogramme pour l'affichage
                hist_max_val = np.max(current_hist) 
                if hist_max_val == 0: hist_max_val = 1 # Éviter la division par zéro

                bin_width_display = hist_display_width / HIST_NUM_BINS
                
                for i in range(HIST_NUM_BINS):
                    bin_height = int((current_hist[i] / hist_max_val) * (hist_display_height - 5)) # -5 pour une petite marge
                    start_x = int(i * bin_width_display)
                    end_x = int((i + 1) * bin_width_display)
                    cv2.rectangle(hist_img, 
                                  (start_x, hist_display_height - bin_height), 
                                  (end_x - 1, hist_display_height -1), # -1 pour séparer les barres
                                  (0, 255, 0), cv2.FILLED) # Barres vertes
            
            # Superposer l'histogramme (ou sa zone vide) sur display_frame
            # Définir la position de l'histogramme (par exemple, en bas)
            hist_y_offset = original_h - hist_display_height - 10 # 10px de marge du bas
            hist_x_offset = 10 # 10px de marge de gauche
            
            # S'assurer que la zone de l'histogramme ne dépasse pas les dimensions de display_frame
            if hist_y_offset + hist_display_height <= original_h and \
               hist_x_offset + hist_display_width <= original_w:
                display_frame[hist_y_offset : hist_y_offset + hist_display_height,
                              hist_x_offset : hist_x_offset + hist_display_width] = hist_img
                cv2.putText(display_frame, "Histo Mvt Horizontal", (hist_x_offset, hist_y_offset - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

            # Afficher une grille (correspondant à la sortie du modèle)
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
