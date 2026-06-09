import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set up visual aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 150})

def load_data(output_dir):
    """Loads all necessary CSVs from the output directory."""
    data = {}
    
    # Load Summary Metrics
    for variant in ['baseline', 'multi']:
        try:
            data[f'conformal_metrics_{variant}'] = pd.read_csv(output_dir / f'conformal_metrics_{variant}.csv')
        except FileNotFoundError:
            pass
            
        try:
            data[f'uq_metrics_{variant}'] = pd.read_csv(output_dir / f'uq_metrics_{variant}.csv')
        except FileNotFoundError:
            pass
            
        # Load Per-Image Inference Data
        try:
            data[f'conformal_test_{variant}'] = pd.read_csv(output_dir / f'conformal_{variant}_test.csv')
        except FileNotFoundError:
            pass
            
        try:
            data[f'mc_dropout_test_{variant}'] = pd.read_csv(output_dir / f'mc_dropout_{variant}_test.csv')
        except FileNotFoundError:
            pass
            
    return data

def plot_calibration(data, output_dir):
    """Plots Target Alpha vs. Observed PICP for both methods and variants."""
    plt.figure(figsize=(8, 6))
    
    # Ideal calibration line
    alphas_ideal = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    plt.plot(alphas_ideal, alphas_ideal, 'k--', label='Ideal Calibration', alpha=0.7)
    
    for variant in ['baseline', 'multi']:
        # Conformal Data
        conf_key = f'conformal_metrics_{variant}'
        if conf_key in data:
            df_conf = data[conf_key]
            # Plot only if we have data points
            if not df_conf.empty:
                plt.plot(df_conf['alpha'], df_conf['PICP'], marker='o', linestyle='-', 
                         label=f'Conformal ({variant.capitalize()})', markersize=8)
                
        # MC Dropout Data
        mc_key = f'uq_metrics_{variant}'
        if mc_key in data:
            df_mc = data[mc_key]
            if not df_mc.empty:
                # Extract alphas and PICPs from columns like 'picp@0.5'
                picp_cols = [c for c in df_mc.columns if c.startswith('picp@')]
                alphas = [float(c.split('@')[1]) for c in picp_cols]
                picps = df_mc.iloc[0][picp_cols].values
                
                # Sort them
                sorted_idx = np.argsort(alphas)
                alphas = np.array(alphas)[sorted_idx]
                picps = np.array(picps)[sorted_idx]
                
                plt.plot(alphas, picps, marker='s', linestyle='-', 
                         label=f'MC Dropout ({variant.capitalize()})', markersize=8)
                
    plt.xlabel('Target Confidence Level (Alpha)')
    plt.ylabel('Observed Coverage (PICP)')
    plt.title('Calibration Curve: Target Alpha vs. Observed Coverage')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'calibration_curve.png')
    plt.close()

def plot_interval_width(data, output_dir):
    """Plots Target Alpha vs. Mean Prediction Interval Width (MPIW)."""
    plt.figure(figsize=(8, 6))
    
    for variant in ['baseline', 'multi']:
        # Conformal Data
        conf_key = f'conformal_metrics_{variant}'
        if conf_key in data:
            df_conf = data[conf_key]
            if not df_conf.empty:
                plt.plot(df_conf['alpha'], df_conf['MPIW'], marker='o', linestyle='-', 
                         label=f'Conformal ({variant.capitalize()})', markersize=8)
                
        # MC Dropout Data
        mc_key = f'uq_metrics_{variant}'
        if mc_key in data:
            df_mc = data[mc_key]
            if not df_mc.empty:
                mpiw_cols = [c for c in df_mc.columns if c.startswith('mpiw@')]
                alphas = [float(c.split('@')[1]) for c in mpiw_cols]
                mpiws = df_mc.iloc[0][mpiw_cols].values
                
                # Sort
                sorted_idx = np.argsort(alphas)
                alphas = np.array(alphas)[sorted_idx]
                mpiws = np.array(mpiws)[sorted_idx]
                
                plt.plot(alphas, mpiws, marker='s', linestyle='-', 
                         label=f'MC Dropout ({variant.capitalize()})', markersize=8)
                
    plt.xlabel('Target Confidence Level (Alpha)')
    plt.ylabel('Mean Prediction Interval Width (Months)')
    plt.title('Prediction Interval Width across Confidence Levels')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'interval_width_curve.png')
    plt.close()

def plot_predictions_scatter(data, output_dir, n_samples=50):
    """Plots Predicted vs True Bone Age with error bars for a random subset."""
    np.random.seed(42) # For reproducibility
    
    for variant in ['baseline', 'multi']:
        conf_key = f'conformal_test_{variant}'
        mc_key = f'mc_dropout_test_{variant}'
        
        has_conf = conf_key in data
        has_mc = mc_key in data
        
        if not (has_conf or has_mc):
            continue
            
        fig, axes = plt.subplots(1, 2 if has_conf and has_mc else 1, figsize=(14 if has_conf and has_mc else 7, 6))
        
        # Ensure axes is iterable
        if not isinstance(axes, np.ndarray):
            axes = [axes]
            
        ax_idx = 0
        
        # Plot Conformal
        if has_conf:
            df = data[conf_key]
            # Sample random points
            if len(df) > n_samples:
                df = df.sample(n_samples, random_state=42)
                
            ax = axes[ax_idx]
            
            # Use 95% intervals if available, else 90%
            alpha_ext = '@0.95' if 'lower@0.95' in df.columns else '@0.9'
            
            y_true = df['true_boneage'].values
            y_pred = df['pred_boneage'].values
            y_lower = df[f'lower{alpha_ext}'].values
            y_upper = df[f'upper{alpha_ext}'].values
            
            # Calculate asymmetric errors for errorbar
            yerr_lower = y_pred - y_lower
            yerr_upper = y_upper - y_pred
            yerr = [yerr_lower, yerr_upper]
            
            ax.errorbar(y_true, y_pred, yerr=yerr, fmt='o', color='blue', 
                        ecolor='lightgray', elinewidth=2, capsize=3, alpha=0.7)
            
            # Plot ideal line
            min_val = min(y_true.min(), y_pred.min()) - 10
            max_val = max(y_true.max(), y_pred.max()) + 10
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
            
            ax.set_xlabel('True Bone Age (Months)')
            ax.set_ylabel('Predicted Bone Age (Months)')
            title_alpha = alpha_ext.replace('@', '')
            ax.set_title(f'Conformal ({variant.capitalize()}) - {title_alpha} PI')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax_idx += 1
            
        # Plot MC Dropout
        if has_mc:
            df = data[mc_key]
            if len(df) > n_samples:
                df = df.sample(n_samples, random_state=42)
                
            ax = axes[ax_idx]
            
            y_true = df['true_boneage'].values
            y_pred = df['pred_mean_boneage'].values
            y_std = df['pred_std_boneage'].values
            
            # 95% PI is approx +/- 1.96 std
            yerr = 1.96 * y_std
            
            ax.errorbar(y_true, y_pred, yerr=yerr, fmt='s', color='green', 
                        ecolor='lightgray', elinewidth=2, capsize=3, alpha=0.7)
            
            # Plot ideal line
            min_val = min(y_true.min(), y_pred.min()) - 10
            max_val = max(y_true.max(), y_pred.max()) + 10
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7)
            
            ax.set_xlabel('True Bone Age (Months)')
            ax.set_ylabel('Predicted Mean Bone Age (Months)')
            ax.set_title(f'MC Dropout ({variant.capitalize()}) - 95% PI (1.96*std)')
            ax.grid(True, linestyle='--', alpha=0.5)
            
        plt.tight_layout()
        plt.savefig(output_dir / 'figures' / f'predictions_scatter_{variant}.png')
        plt.close()

def plot_error_histogram(data, output_dir):
    """Plots histogram of absolute errors across variants."""
    plt.figure(figsize=(10, 6))
    
    for variant in ['baseline', 'multi']:
        conf_key = f'conformal_test_{variant}'
        
        if conf_key in data:
            df = data[conf_key]
            sns.kdeplot(df['abs_error'], label=f'{variant.capitalize()} Model', 
                        fill=True, alpha=0.4, linewidth=2)
            
    plt.xlabel('Absolute Error (Months)')
    plt.ylabel('Density')
    plt.title('Distribution of Absolute Errors')
    plt.xlim(0, 50) # Clip long tail for better visualization
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'absolute_error_distribution.png')
    plt.close()

def main():
    _root = Path(__file__).resolve().parent.parent.parent
    output_dir = Path(os.getenv('BONE_AGE_OUTPUT_DIR', str(_root / 'tf-pediatric-bone-age/outputs/rsna_boneage_models')))
    
    # Create figures directory
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading data from {output_dir}")
    data = load_data(output_dir)
    
    if not data:
        print("No CSV data found. Please run inference scripts first.")
        return
        
    print("Generating Calibration Curve...")
    plot_calibration(data, output_dir)
    
    print("Generating Interval Width Plot...")
    plot_interval_width(data, output_dir)
    
    print("Generating Prediction Scatter Plots...")
    plot_predictions_scatter(data, output_dir)
    
    print("Generating Absolute Error Histogram...")
    plot_error_histogram(data, output_dir)
    
    print(f"All figures saved to {figures_dir}")

if __name__ == '__main__':
    main()
