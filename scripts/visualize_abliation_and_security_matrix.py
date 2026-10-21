import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set global aesthetic style matching the output images
sns.set_theme(style="whitegrid")

# ==============================================================================
# 1. ABLATION METRICS VISUALIZATION (2x2 Grid)
# ==============================================================================

def compute_ablation_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Accuracy, False Authorization Rate (FAR), Unsafe Action Block Rate (UABR),
    and Safe Action Pass Rate (SAPR) grouped by variant and attack_category.
    """
    results = []
    
    for (variant, category), group in df.groupby(['variant', 'attack_category']):
        # Overall Accuracy
        accuracy = (group['predicted_decision'] == group['expected_decision']).mean()
        
        # Unsafe scenarios (where expected_decision is 'BLOCK')
        unsafe_mask = group['expected_decision'] == 'BLOCK'
        if unsafe_mask.sum() > 0:
            far = ((group['predicted_decision'] == 'ALLOW') & unsafe_mask).sum() / unsafe_mask.sum()
            uabr = ((group['predicted_decision'] == 'BLOCK') & unsafe_mask).sum() / unsafe_mask.sum()
        else:
            far = np.nan
            uabr = np.nan
            
        # Safe scenarios (where expected_decision is 'ALLOW')
        safe_mask = group['expected_decision'] == 'ALLOW'
        if safe_mask.sum() > 0:
            sapr = ((group['predicted_decision'] == 'ALLOW') & safe_mask).sum() / safe_mask.sum()
        else:
            sapr = np.nan
            
        results.append({
            'variant': variant,
            'attack_category': category,
            'Accuracy': accuracy,
            'False Authorization Rate': far,
            'Uabr': uabr,
            'Sapr': sapr
        })
        
    return pd.DataFrame(results)

def plot_ablation_metrics(df_raw: pd.DataFrame, output_filename: str = "ablation_metrics.png"):
    """Generates the 2x2 grouped bar plot grid for ablation evaluations."""
    df_metrics = compute_ablation_metrics(df_raw)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    configs = [
        ('Accuracy', 'Overall Accuracy (Higher is better)', 'Accuracy', axes[0, 0]),
        ('False Authorization Rate', 'False Authorization Rate (Lower is better)', 'False Authorization Rate', axes[0, 1]),
        ('Uabr', 'Unsafe Action Block Rate - UABR (Higher is better)', 'Uabr', axes[1, 0]),
        ('Sapr', 'Safe Action Pass Rate - SAPR (Higher is better)', 'Sapr', axes[1, 1])
    ]
    
    for idx, (metric_col, title, ylabel, ax) in enumerate(configs):
        sns.barplot(
            data=df_metrics,
            x='attack_category',
            y=metric_col,
            hue='variant',
            ax=ax
        )
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("Attack Category", fontsize=11, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_ylim(0.0, 1.05)
        ax.tick_params(axis='x', rotation=25)
        
        # Only retain the legend on the first subplot to prevent plot redundancy
        if idx > 0 and ax.get_legend() is not None:
            ax.get_legend().remove()
            
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================================================
# 2. FEDERATED LEARNING SECURITY METRICS VISUALIZATION (1x2 Grid)
# ==============================================================================

def plot_fl_security_metrics(df_fl: pd.DataFrame, output_filename: str = "fl_security_metrics.png"):
    """Generates the 1x2 line plot grid for FL security evaluations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Subplot 1: Model Performance vs. Attacker Presence
    sns.lineplot(
        data=df_fl,
        x='malicious_clients_pct',
        y='macro_f1',
        hue='mode',
        style='attack',
        markers=True,
        dashes=False,
        ax=axes[0]
    )
    axes[0].set_title("Model Performance (Macro F1) vs. Attacker Presence", fontsize=12, pad=10)
    axes[0].set_xlabel("Malicious Clients (%)", fontsize=11, labelpad=10)
    axes[0].set_ylabel("Macro F1 Score", fontsize=11)
    
    # Subplot 2: Malicious Client Detection Rate
    sns.lineplot(
        data=df_fl,
        x='malicious_clients_pct',
        y='malicious_client_detection_rate',
        hue='mode',
        style='attack',
        markers=True,
        dashes=False,
        ax=axes[1]
    )
    axes[1].set_title("Defense Efficacy: Malicious Client Detection Rate", fontsize=12, pad=10)
    axes[1].set_xlabel("Malicious Clients (%)", fontsize=11, labelpad=10)
    axes[1].set_ylabel("Detection Rate", fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()


# ==============================================================================
# MAIN EXECUTION PIPELINE
# ==============================================================================

if __name__ == "__main__":
    # Load raw CSV data
    df_component = pd.read_csv("artifacts/astb_benchmark/component_ablation_rows.csv")
    df_fl_security = pd.read_csv("artifacts/integrity/fl_security_metrics.csv")
    
    # Generate Plots
    plot_ablation_metrics(df_component, output_filename="ablation_metrics_plot.png")
    plot_fl_security_metrics(df_fl_security, output_filename="fl_security_metrics_plot.png")