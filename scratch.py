import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_progression_formal(cost_list, val_iou_list, train_iou_list, model_name):
    data = {'Epoch': list(range(len(cost_list))),
            'Validation IoU': val_iou_list,
            'Training IoU': train_iou_list,
            'Training Loss': cost_list}
    df = pd.DataFrame(data)

    # Create a figure and axis
    fig, (ax1, ax_capt) = plt.subplots(nrows=2, figsize=(10, 7), gridspec_kw={'height_ratios': [6, 1]})
    sns.set_style("whitegrid") # Set seaborn style

    # Plot the IoU data on the first axis
    line1 = sns.lineplot(x='Epoch', y='value', hue='variable', palette=['tab:red', 'tab:orange'], data=pd.melt(df, ['Epoch'], value_vars=['Validation IoU', 'Training IoU']), markers=["o", "x"], dashes=False, ax=ax1)
    ax1.set_ylabel('IoU')

    ax2 = ax1.twinx()
    line2, = sns.lineplot(x='Epoch', y='Training Loss', data=df, color='tab:blue', ax=ax2).get_lines()
    ax2.set_ylabel('Training loss')

    plt.title('Loss and Accuracy Progression for {}'.format(model_name))

    # Create legend manually
    lines, labels = ax1.get_legend_handles_labels()
    lines.append(line2)
    labels.append('Training Loss')
    ax1.legend(lines, labels[:], title="Metric", loc='upper left', bbox_to_anchor=(1.07, 1)) # Outside the plot

    plt.tight_layout()

    # Add caption
    ax_capt.axis('off') # Turn off the axis for the caption
    ax_capt.text(0.5, 0.5, 'Caption goes here', ha='center') # Add centered caption text

    # Save the figure
    plt.savefig('/home/mason2/AGVon1080Ti/FormalProg{}.png'.format(model_name), bbox_inches='tight', format='png')

    plt.close()

# Example usage
cost_list = [0.5, 0.4, 0.3]
val_iou_list = [0.6, 0.7, 0.8]
train_iou_list = [0.5, 0.6, 0.7]
model_name = "Model5"
plot_progression_formal(cost_list, val_iou_list, train_iou_list, model_name)
