import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Calculates distribution of every class in the y_train dataset
# e.g: 12% of c01, 8.5% of c02, etc...
def class_distribution(y_train, threshold) -> pd.DataFrame:
    n_samples = len(y_train)

    # vector operations
    mask = y_train >= threshold
    counts = mask.sum(axis=0)
    percent = (counts / n_samples) * 100


    df = pd.DataFrame({
        "class": percent.index,
        "percentage": percent.values 
        })

    return df

# Generates a chart for the class distribution
def distribution_visualization(df: pd.DataFrame, threshold: float):
    plt.figure(figsize=(8,4))
    plt.bar(df["class"], df["percentage"], color="skyblue")
    plt.title(f"Distribution des classes")
    plt.xlabel("Classes")
    plt.ylabel("Taux de présence")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    #plt.savefig("class_distribution.pdf", bbox_inches='tight')

def sensors_heatmap(x_train: pd.DataFrame):
    plt.figure(figsize=(12, 10))
    correlation = x_train.corr()
    sns.heatmap(correlation, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.gca().invert_yaxis()
    plt.title('Corrélation des capteurs')
    plt.tight_layout()
    plt.show()
    #plt.savefig("heatmap.pdf", bbox_inches='tight')

# Shows an evolution of the sensors values to see synchronization
def plot_multi_sensors_dashboard(x_train, start_idx=30, end_idx=50):
    sensors_to_plot = ['M4', 'M5', 'S1', 'R', 'Humidity'] # Arbitrary chosen sensors to show
    
    df_subset = x_train.sort_values('ID').iloc[start_idx:end_idx]
    ids = df_subset['ID']
    
    fig, axes = plt.subplots(len(sensors_to_plot), 1, figsize=(12, 10), sharex=True)
    
    for i, sensor in enumerate(sensors_to_plot):
        ax = axes[i]
        ax.plot(ids, df_subset[sensor], label=sensor, color=f'C{i}')
        
        ax.set_ylabel(sensor)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # To fill under the curve area
        ax.fill_between(ids, df_subset[sensor], alpha=0.1, color=f'C{i}')

    plt.xlabel("ID")
    plt.suptitle(f"ID Window {start_idx} à {end_idx}", fontsize=16)
    plt.tight_layout()
    plt.show()

# Function used to show an example of an entry/output of the dataset
def dataset_preview(df_train, df_test):
    train_sample = df_train.head(5).round(3)     
    test_sample = df_test.head(5).round(3)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 6))
    plt.subplots_adjust(hspace=0.5)

    def render_table(ax, df, title, color_header):
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)

        # Colors
        for (row, col), cell in table.get_celld().items():
            if row == 0: # Header
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor(color_header)
                cell.set_edgecolor('white')
            else: # Data
                cell.set_facecolor('#f2f2f2' if row % 2 else 'white')
                cell.set_edgecolor('#dddddd')
        
        ax.set_title(title, fontsize=14, weight='bold', loc='left', color=color_header)

    render_table(ax1, train_sample, "Sample entrées", "#40466e")
    render_table(ax2, test_sample, "Sample sorties", "#e67e22")

    plt.show()
    #plt.savefig("dataset_preview.pdf", bbox_inches='tight')

def show_graphs(x_train, y_train, x_test):
    distribution_visualization(class_distribution(y_train.drop(columns=["ID"]), 0.50), 0.50)
    sensors_heatmap(x_train.drop(columns=["ID"]))
    plot_multi_sensors_dashboard(x_train)
    dataset_preview(x_train, x_test)
