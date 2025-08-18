import os
import pandas as pd
import numpy as np
import argparse

def analyze_movement_data(reports_directory):
    """
    Analyse les données de mouvement X et Y à partir des fichiers CSV
    dans le répertoire spécifié.
    Calcule et affiche le max, le 95ème percentile, la moyenne et l'écart-type.
    """
    all_move_x = []
    all_move_y = []
    
    if not os.path.isdir(reports_directory):
        print(f"Erreur: Le répertoire '{reports_directory}' n'existe pas.")
        return

    print(f"Analyse des fichiers CSV dans le répertoire : {reports_directory}")
    
    csv_files_found = 0
    for filename in os.listdir(reports_directory):
        if filename.lower().endswith('_movement.csv'):
            csv_files_found += 1
            file_path = os.path.join(reports_directory, filename)
            try:
                df = pd.read_csv(file_path)
                if 'Move_X' in df.columns and 'Move_Y' in df.columns:
                    all_move_x.extend(df['Move_X'].dropna().tolist())
                    all_move_y.extend(df['Move_Y'].dropna().tolist())
                else:
                    print(f"Avertissement: Les colonnes 'Move_X' ou 'Move_Y' sont manquantes dans {filename}")
            except pd.errors.EmptyDataError:
                print(f"Avertissement: Le fichier CSV {filename} est vide.")
            except Exception as e:
                print(f"Erreur lors de la lecture du fichier {filename}: {e}")

    if csv_files_found == 0:
        print(f"Aucun fichier CSV se terminant par '_movement.csv' n'a été trouvé dans '{reports_directory}'.")
        return

    if not all_move_x and not all_move_y:
        print("Aucune donnée de mouvement n'a été extraite des fichiers CSV.")
        return

    print(f"\nNombre total de points de données Move_X extraits : {len(all_move_x)}")
    print(f"Nombre total de points de données Move_Y extraits : {len(all_move_y)}")

    if all_move_x:
        move_x_np = np.array(all_move_x)
        print("\n--- Statistiques pour Move_X ---")
        print(f"Maximum: {np.max(move_x_np):.2f}")
        print(f"95ème percentile: {np.percentile(move_x_np, 95):.2f}")
        print(f"Moyenne: {np.mean(move_x_np):.2f}")
        print(f"Écart-type: {np.std(move_x_np):.2f}")
    else:
        print("\n--- Statistiques pour Move_X ---")
        print("Aucune donnée Move_X à analyser.")

    if all_move_y:
        move_y_np = np.array(all_move_y)
        print("\n--- Statistiques pour Move_Y ---")
        print(f"Maximum: {np.max(move_y_np):.2f}")
        print(f"95ème percentile: {np.percentile(move_y_np, 95):.2f}")
        print(f"Moyenne: {np.mean(move_y_np):.2f}")
        print(f"Écart-type: {np.std(move_y_np):.2f}")
    else:
        print("\n--- Statistiques pour Move_Y ---")
        print("Aucune donnée Move_Y à analyser.")

def main():
    parser = argparse.ArgumentParser(description="Analyse la distribution des données de mouvement X et Y à partir des rapports CSV.")
    parser.add_argument(
        'reports_directory', 
        type=str, 
        nargs='?', 
        default="camera_movement_reports",
        help="Répertoire contenant les fichiers CSV des rapports de mouvement (par défaut: 'camera_movement_reports')."
    )
    args = parser.parse_args()

    analyze_movement_data(args.reports_directory)

if __name__ == "__main__":
    main()
