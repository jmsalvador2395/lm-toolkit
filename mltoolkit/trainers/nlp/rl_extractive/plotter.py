from datasets import Dataset
from matplotlib import pyplot as plt
import numpy as np

def plot_rouge_history(hist, mode):

    hist = Dataset.from_list(hist)

    x = hist['step']
    plots = {}

    metrics = ['rouge1', 'rouge2', 'rougeL']
    subcats = ['fmeasure', 'precision', 'recall']
    alphas = [.5, .8, .5]
    styles = ['r-', 'b--', 'g-.']
    markers = ['.', '.', '.']

    for metric in metrics:

        fig, ax = setup_plot()
        for cat, alpha, style, marker in zip(subcats, alphas, styles, markers):
            ax.plot(
                x,
                hist[f'{metric}-{cat}'],
                style,
                alpha=alpha,
                marker=marker,
                label=cat
            )

        plt.title(f'{metric}')
        plt.subplots_adjust(left=.15, right=.9, bottom=.15, top=.9)
        legend = ax.legend(loc='best', shadow=True)
        plt.xlabel('Training Step')

        #xticks = [human_format(num) for num in x]

        # dump into numpy array
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close()
        
        plots[f'{metric}/{mode}'] = img
    
    return plots

def plot_reward_history(hist, mode):

    # setup vars
    hist = Dataset.from_list(hist)
    x = hist['step']

    metrics = ['aggregate', 'base term', 'budget term']
    alphas = [.5, .8, .5]
    styles = ['r-', 'b--', 'g-.']
    markers = ['.', '.', '.']

    fig, ax = setup_plot()
    for metric, alpha, style, marker in zip(metrics, alphas, styles, markers):
        ax.plot(
            x, 
            hist[metric],
            style,
            alpha=alpha,
            marker=marker,
            label=metric
        )

    plt.subplots_adjust(left=.15, right=.9, bottom=.15, top=.9)
    legend = ax.legend(loc='best', shadow=True)
    plt.title('Average Reward')
    plt.xlabel('Training Step')

    # plot and save to image
    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()

    return {
        f'reward/{mode}': img
    }

def setup_plot():

    w=8
    h=6
    font_size=20
    legend_font_size=16
    plt.rc('font', size=font_size)
    plt.rc('legend', fontsize=legend_font_size)
    plt.rc('axes', titlesize=font_size+4)
    plt.rc('axes', labelsize=font_size+4)
    plt.rc('xtick', labelsize=font_size-2)
    plt.rc('ytick', labelsize=font_size-2)

    fig, ax = plt.subplots(figsize=(w,h))
    ax.grid()

    return fig, ax

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
