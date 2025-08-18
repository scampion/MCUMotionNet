import cv2
import os
import numpy as np
import csv
import argparse

def estimate_camera_movement(video_path, debug_video_output_path=None):
    """
    Estime le mouvement de la caméra dans une vidéo en utilisant le flux optique.
    Génère optionnellement une vidéo de débogage avec les mouvements affichés.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la vidéo {video_path}")
        return None

    ret, prev_frame_bgr = cap.read()
    if not ret:
        print(f"Erreur: Impossible de lire la première image de {video_path}")
        cap.release()
        return None
    
    old_gray = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
    
    out_video = None
    if debug_video_output_path:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: # Handle case where FPS might not be readable or is zero
            fps = 25 # Default FPS
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec
        out_video = cv2.VideoWriter(debug_video_output_path, fourcc, fps, (frame_width, frame_height))
        
        # Écrire la première image dans la vidéo de débogage (sans annotations de mouvement)
        if out_video and prev_frame_bgr is not None:
            out_video.write(prev_frame_bgr)

    # Paramètres pour la détection de points d'intérêt (ShiTomasi)
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    
    # Paramètres pour le flux optique (Lucas-Kanade)
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    frame_count = 0 # La première image est déjà lue (prev_frame_bgr)
    # significant_movement_frames = 0 # Supprimé car nous stockons les détails image par image
    detailed_frame_movements = []

    while True:
        ret, current_frame_bgr = cap.read()
        if not ret:
            break # Fin de la vidéo

        frame_count += 1 # On compte à partir de la deuxième image traitée dans la boucle
        current_frame_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)

        if p0 is not None and len(p0) > 0:
            # Calculer le flux optique
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, current_frame_gray, p0, None, **lk_params)

            # Sélectionner les bons points
            good_new = None
            good_old = None
            if st is not None:
                good_new = p1[st==1]
                good_old = p0[st==1]

            # Calculer le mouvement moyen
            if good_new is not None and len(good_new) > 0 and good_old is not None and len(good_old) > 0:
                move_x = np.mean(good_new[:,0] - good_old[:,0])
                move_y = np.mean(good_new[:,1] - good_old[:,1])
                
                magnitude = np.sqrt(move_x**2 + move_y**2)
                
                # Stocker et imprimer les détails du mouvement image par image
                detailed_frame_movements.append((frame_count, move_x, move_y, magnitude))
                print(f"Vidéo: {os.path.basename(video_path)}, Image: {frame_count}, Mouvement X: {move_x:.2f}, Mouvement Y: {move_y:.2f}, Magnitude: {magnitude:.2f}")
                
                if out_video:
                    # Dessiner les informations sur l'image actuelle (current_frame_bgr)
                    # Il est important de noter que move_x, move_y, magnitude sont calculés à partir de la différence entre old_gray et current_frame_gray
                    # Donc, ces valeurs représentent le mouvement qui a conduit à current_frame_bgr
                    display_frame = current_frame_bgr.copy() # Travailler sur une copie pour ne pas affecter current_frame_gray
                    cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Move X: {move_x:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Move Y: {move_y:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Magnitude: {magnitude:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    out_video.write(display_frame)
            elif out_video: # Si p0 est None ou pas de bons points, écrire l'image sans annotations de mouvement
                 display_frame = current_frame_bgr.copy()
                 cv2.putText(display_frame, f"Frame: {frame_count} (No trackable points or movement)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                 out_video.write(display_frame)

        
        # Mettre à jour l'image précédente
        old_gray = current_frame_gray.copy()
        # Mettre à jour les points pour le suivi (ou les re-détecter)
        # Pour la robustesse, il est souvent préférable de re-détecter les points périodiquement
        # ou lorsque le nombre de points suivis devient trop faible.
        # Le script fourni re-détecte à chaque image.
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)


    cap.release()
    if out_video:
        out_video.release()

    # frame_count ici représente le nombre d'itérations de la boucle, 
    # donc le nombre d'images après la première.
    # Le nombre total d'images est frame_count + 1 (si la première image a été lue avec succès)
    # ou simplement frame_count si on considère le nombre de transitions analysées.
    # Pour être cohérent avec l'ancienne méthode qui comptait toutes les images lues dans la boucle :
    # Si la vidéo a N images, la boucle s'exécute N-1 fois.
    # L'ancienne méthode comptait N images si N>0.
    # Ajustons frame_count pour qu'il représente le nombre total d'images traitées pour le flux optique.
    # Si la première image est lue (ret=True pour prev_frame_bgr), et la boucle tourne F fois,
    # alors F+1 images ont été impliquées.
    # La variable frame_count est initialisée à 0 et incrémentée dans la boucle.
    # Si la vidéo a 1 image, la boucle ne tourne pas, frame_count = 0.
    # Si la vidéo a 2 images, la boucle tourne 1 fois, frame_count = 1.
    # Le nombre total d'images lues est cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # Mais nous nous intéressons au nombre d'images où le mouvement a pu être calculé.
    # Le frame_count actuel est le nombre de paires d'images analysées.
    # Pour que le "Total images" dans le CSV soit plus intuitif (nombre total d'images dans la vidéo),
    # il faudrait le récupérer avant. Mais pour le pourcentage, il faut le nombre d'analyses.
    # Gardons la logique actuelle où frame_count est le nombre de transitions analysées.
    # Si la vidéo a au moins 2 images, frame_count sera > 0.
    
    # La logique de total_frames_analyzed n'est plus centrale pour la sortie.
    # La fonction retourne maintenant tous les mouvements détaillés.

    # L'impression récapitulative qui était ici est supprimée, car les données image par image sont imprimées ci-dessus.
    # L'ancienne signature de retour est modifiée.
    return os.path.basename(video_path), detailed_frame_movements


def main():
    parser = argparse.ArgumentParser(description="Calcule le mouvement de caméra des vidéos et génère des rapports CSV.")
    parser.add_argument('--create-debug-videos', action='store_true',
                        help="Générer des vidéos de débogage avec les informations de mouvement superposées.")
    args = parser.parse_args()

    video_directory = "data/videos"
    output_reports_directory = "camera_movement_reports"
    debug_videos_output_directory = "data/videos_with_debug_overlay" # Nouveau répertoire pour les vidéos de débogage
    
    if not os.path.exists(video_directory):
        print(f"Erreur: Le répertoire '{video_directory}' n'existe pas.")
        return

    # Création du répertoire pour les rapports CSV
    os.makedirs(output_reports_directory, exist_ok=True)

    if args.create_debug_videos:
        os.makedirs(debug_videos_output_directory, exist_ok=True) # Créer le répertoire pour les vidéos de débogage

    supported_extensions = ('.mp4', '.avi', '.mov', '.mkv') # Ajoutez d'autres extensions si nécessaire

    processed_files_count = 0

    for filename in os.listdir(video_directory):
        if filename.lower().endswith(supported_extensions):
            video_path = os.path.join(video_directory, filename)
            video_basename_no_ext = os.path.splitext(filename)[0]
            
            current_debug_video_path = None
            if args.create_debug_videos:
                current_debug_video_path = os.path.join(debug_videos_output_directory, f"{video_basename_no_ext}_debug.mp4")

            video_basename = os.path.basename(video_path)
            csv_filename_base = os.path.splitext(video_basename)[0]
            output_csv_path = os.path.join(output_reports_directory, f"{csv_filename_base}_movement.csv")
            if os.path.exists(output_csv_path):
                print(f"Already processed: {output_csv_path}, skipping.")
                continue

            print(f"Traitement de la vidéo : {video_path}")
            result_tuple = estimate_camera_movement(video_path, debug_video_output_path=current_debug_video_path)
            
            if result_tuple:
                video_basename, frame_movements = result_tuple
                
                if not frame_movements: 
                    print(f"Aucun mouvement calculable pour {video_basename}, CSV non généré.")
                else:

                    try:
                        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                            csv_writer = csv.writer(csvfile)
                            # Écriture de l'en-tête
                            csv_writer.writerow(['Frame', 'Move_X', 'Move_Y', 'Magnitude'])
                            # Écriture des données pour cette vidéo (frame par frame)
                            for frame_data in frame_movements:
                                csv_writer.writerow(frame_data)
                        print(f"Les détails de mouvement pour {video_basename} ont été sauvegardés dans {output_csv_path}")
                        processed_files_count += 1
                    except IOError:
                        print(f"Erreur: Impossible d'écrire dans le fichier CSV {output_csv_path}")
            else:
                # Ce cas gère si estimate_camera_movement a retourné None (par ex., erreur d'ouverture vidéo)
                print(f"Le traitement a échoué pour {video_path} (estimate_camera_movement a retourné None).")
        else:
            print(f"Ignoré (pas une vidéo supportée): {filename}")

    if processed_files_count == 0:
        print("Aucune vidéo n'a été traitée ou aucun résultat n'a été généré.")
    else:
        print(f"Traitement terminé. {processed_files_count} rapports CSV ont été générés dans '{output_reports_directory}'.")


if __name__ == "__main__":
    main()
