import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_professional_figure(csv_path):
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({'font.size': 11, 'pdf.fonttype': 42})

    print("Génération de la Figure : Lignes ajoutées vs supprimées (RQ4)...")
    df_mod = df[df['backport_type'] == 'Modified'].copy()
    
    df_mod['delta_added'] = df_mod['stable_lines_added'] - df_mod['master_lines_added']
    df_mod['delta_removed'] = df_mod['stable_lines_removed'] - df_mod['master_lines_removed']
    
    n = len(df_mod)
    added = [len(df_mod[df_mod['delta_added'] > 0])/n*100, 
             len(df_mod[df_mod['delta_added'] < 0])/n*100, 
             len(df_mod[df_mod['delta_added'] == 0])/n*100]
    
    removed = [len(df_mod[df_mod['delta_removed'] > 0])/n*100, 
               len(df_mod[df_mod['delta_removed'] < 0])/n*100, 
               len(df_mod[df_mod['delta_removed'] == 0])/n*100]

    labels =['More than Master', 'Less than Master', 'Same as Master']
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    color_added = '#4C72B0' 
    color_removed = '#8C92AC'
    
    rects1 = ax.bar(x - width/2, added, width, label='Lines Added', color=color_added, edgecolor='black', linewidth=0.8)
    rects2 = ax.bar(x + width/2, removed, width, label='Lines Removed', color=color_removed, edgecolor='black', linewidth=0.8)

    ax.set_ylabel('Percentage of Modified Backports (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 50)

    for rects in[rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('rq4_lines_added_removed.pdf')
    plt.close()
    print("Figure 'rq4_lines_added_removed.pdf' générée avec succès.")

if __name__ == "__main__":
    generate_professional_figure("all_backports_openstack_with_backport_type.csv")