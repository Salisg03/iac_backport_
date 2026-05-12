import pandas as pd
import zipfile
import json
import os

def calculate_rq1_prevalence(backports_csv, all_changes_zip):
    print("1. Lecture des backports IaC...")
    df_bp = pd.read_csv(backports_csv)
    
    # Projets IaC uniques
    iac_projects = set(df_bp['project'].dropna().unique())
    backported_master_patches = df_bp['change_id'].nunique()
    total_backport_instances = len(df_bp)
    
    print(f"-> Projets IaC identifiés : {len(iac_projects)}")
    print(f"-> Patchs originaux backportés : {backported_master_patches}")
    print(f"-> Total instances (master-stable) : {total_backport_instances}\n")

    print(f"2. Analyse du fichier JSON ({all_changes_zip}) ligne par ligne...")
    
    total_master_merged = 0
    total_master_merged_in_iac_projects = 0
    lines_processed = 0
    
    with zipfile.ZipFile(all_changes_zip, 'r') as z:
        file_list = z.namelist()
        data_file = file_list[0]
        print(f"   -> Fichier trouvé dans l'archive : {data_file}")
        
        with z.open(data_file) as f:
            for line in f:
                lines_processed += 1
                try:
                    # Décoder la ligne et la convertir en dictionnaire JSON
                    change = json.loads(line.decode('utf-8'))
                    
                    # Extraire les champs nécessaires (avec get pour éviter les erreurs si la clé n'existe pas)
                    branch = change.get('branch')
                    status = change.get('status')
                    project = change.get('project')
                    
                    # On cherche les patchs fusionnés sur 'master'
                    if branch == 'master' and status == 'MERGED':
                        total_master_merged += 1
                        
                        # Vérifier si c'est un des projets IaC étudiés
                        if project in iac_projects:
                            total_master_merged_in_iac_projects += 1
                            
                except json.JSONDecodeError:
                    # Ignore les lignes vides ou mal formatées
                    pass
                    
                if lines_processed % 100000 == 0:
                    print(f"   ... {lines_processed} changements analysés ...")

    
    print("RÉSULTATS FINAUX POUR LA RQ1 ET LE DATASET")
    print(f"Total EXACT des changements (toutes branches/statuts) dans le zip : {lines_processed}")
    print(f"Total des patchs fusionnés sur 'master' (tous projets OpenStack) : {total_master_merged}")
    print(f"Total des patchs fusionnés sur 'master' (dans nos {len(iac_projects)} projets IaC) : {total_master_merged_in_iac_projects}")
    print(f"Total des patchs originaux IaC backportés : {backported_master_patches}")
    
    if total_master_merged_in_iac_projects > 0:
        prevalence = (backported_master_patches / total_master_merged_in_iac_projects) * 100
        print(f"\n=> PRÉVALENCE DU BACKPORTING (dans ces projets) : {prevalence:.2f}%")
    else:
        print("\n=> Impossible de calculer le pourcentage (dénominateur = 0).")
    
if __name__ == "__main__":
    backports_file = "all_backports_openstack_with_backport_type.csv"
    zip_file = "opendev_all_changes.zip"
    
    if os.path.exists(backports_file) and os.path.exists(zip_file):
        calculate_rq1_prevalence(backports_file, zip_file)
    else:
        print(f"Erreur : '{backports_file}' ou '{zip_file}' introuvable.")