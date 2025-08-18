import cv2
import os
import numpy as np
import argparse

def display_video_with_movement(video_path):
    """
    Affiche une vidéo avec la visualisation du mouvement de la caméra en temps réel.
    Le mouvement (X, Y, Magnitude) affiché sur une image N correspond au mouvement
    calculé entre l'image N-1 et l'image N.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la vidéo {video_path}")
        return

    # Lire la première image
    ret, frame_bgr = cap.read()
    if not ret:
        print(f"Erreur: Impossible de lire la première image de {video_path}")
        cap.release()
        return
    
    old_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # Paramètres pour la détection de points d'intérêt (ShiTomasi)
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )
    
    # Paramètres pour le flux optique (Lucas-Kanade)
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Détecter les points d'intérêt sur la première image
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    
    frame_number = 1 # Numéro de l'image actuelle (1-indexé)

    # Afficher la première image (pas de mouvement calculé pour elle)
    display_frame_init = frame_bgr.copy()
    cv2.putText(display_frame_init, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display_frame_init, "Move X: N/A", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display_frame_init, "Move Y: N/A", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display_frame_init, "Magnitude: N/A", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow('Video Movement Visualizer', display_frame_init)

    while True:
        ret, frame_bgr = cap.read() # Lire l'image suivante
        if not ret:
            break # Fin de la vidéo

        frame_number += 1
        current_frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        display_frame = frame_bgr.copy() # Image sur laquelle dessiner les informations

        if p0 is not None and len(p0) > 0:
            # Calculer le flux optique
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, current_frame_gray, p0, None, **lk_params)

            good_new = None
            good_old = None
            if st is not None: # st peut être None si p0 est vide après un filtrage potentiel
                good_new = p1[st==1]
                good_old = p0[st==1]

            # Calculer le mouvement moyen
            if good_new is not None and len(good_new) > 0 and good_old is not None and len(good_old) > 0:
                move_x = np.mean(good_new[:,0] - good_old[:,0])
                move_y = np.mean(good_new[:,1] - good_old[:,1])
                magnitude = np.sqrt(move_x**2 + move_y**2)
                
                cv2.putText(display_frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Move X: {move_x:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Move Y: {move_y:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Magnitude: {magnitude:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else: # Pas de bons points suivis
                cv2.putText(display_frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, "Move X: N/A (no track)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, "Move Y: N/A (no track)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(display_frame, "Magnitude: N/A (no track)", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else: # p0 était None ou vide (pas de points d'intérêt initiaux)
            cv2.putText(display_frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(display_frame, "Move X: N/A (no features)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(display_frame, "Move Y: N/A (no features)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(display_frame, "Magnitude: N/A (no features)", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('Video Movement Visualizer', display_frame)
        
        # Mettre à jour l'image précédente en niveaux de gris
        old_gray = current_frame_gray.copy()
        # Redétecter les points d'intérêt pour l'image suivante
        # Pour la robustesse, il est souvent préférable de re-détecter les points périodiquement.
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

        # Attendre une touche, 'q' pour quitter.
        # Le délai affecte la vitesse de lecture. fps = cap.get(cv2.CAP_PROP_FPS); delay = int(1000/fps)
        delay = 30 # ms, environ 33 FPS.
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Visualise le mouvement de la caméra d'une vidéo en temps réel.")
    parser.add_argument('video_path', type=str, help="Chemin vers le fichier vidéo à analyser.")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Erreur: Le fichier vidéo '{args.video_path}' n'existe pas.")
        return
    
    if not os.path.isfile(args.video_path):
        print(f"Erreur: Le chemin '{args.video_path}' n'est pas un fichier valide.")
        return

    display_video_with_movement(args.video_path)

if __name__ == "__main__":
    main()
