import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_extreme_cases(csv_path):
    df = pd.read_csv(csv_path)
    total = len(df)
    print("1. POURCENTAGE DES CAS 'PIRES QUE MASTER'")
    
    harder_time = len(df[df['bp_review_hours'] > df['master_review_hours']])
    harder_patchsets = len(df[df['bp_patchsets'] > df['master_patchsets']])
    harder_ci = len(df[df['bp_ci_failures'] > df['master_ci_failures']])
    harder_msgs = len(df[df['bp_human_messages'] > df['master_human_messages']])
    
    print(f"Review Hours : {harder_time} cas ({harder_time/total*100:.2f}%)")
    print(f"Comments     : {harder_msgs} cas ({harder_msgs/total*100:.2f}%)")
    print(f"Patchsets    : {harder_patchsets} cas ({harder_patchsets/total*100:.2f}%)")
    print(f"CI Failures  : {harder_ci} cas ({harder_ci/total*100:.2f}%)")

    # Génération de la Figure
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42})

    metrics =['Review Duration', 'Human Comments', 'Patchsets', 'CI Failures']
    percentages =[harder_time/total*100, harder_msgs/total*100, harder_patchsets/total*100, harder_ci/total*100]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    # Couleur rouge brique pour montrer la "douleur/difficulté"
    bars = ax.barh(metrics, percentages, color="#4C72B0", edgecolor='black', linewidth=0.8)
    
    ax.set_xlabel('Percentage of Backports (%)')
    ax.set_title('Frequency of Backports Requiring MORE Effort Than Master')
    ax.set_xlim(0, max(percentages) + 10)

    # Ajouter le % au bout de chaque barre
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', ha='left', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('rq3_harder_than_master.pdf')
    print("\n=> Figure 'rq3_harder_than_master.pdf' générée avec succès !")

    print("2. LE TABLEAU DES CAS EXTRÊMES (Top 10% les plus difficiles)")
    
    # On isole les backports modifiés
    df_mod = df[df['backport_type'] == 'Modified'].copy()
    
    # On prend le top 10% basé sur le temps de revue
    p90_mod_hours = df_mod['bp_review_hours'].quantile(0.90)
    hell_cases = df_mod[df_mod['bp_review_hours'] >= p90_mod_hours]
    
    print(f"Statistiques médianes pour le Top 10% des Modified Backports (N={len(hell_cases)}):")
    print(f"- Review Hours : {hell_cases['bp_review_hours'].median():.2f}")
    print(f"- Patchsets    : {hell_cases['bp_patchsets'].median():.2f}")
    print(f"- Comments     : {hell_cases['bp_human_messages'].median():.2f}")
    print(f"- CI Failures  : {hell_cases['bp_ci_failures'].median():.2f}")

if __name__ == "__main__":
    generate_extreme_cases("all_backports_openstack_with_backport_type.csv")