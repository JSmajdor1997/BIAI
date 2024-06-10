from typing import List, Tuple

import matplotlib.pyplot as plt

class PlotConfig:
    def __init__(self, name: str, color: str):
        self.name = name
        self.color = color


def getPlotter(title: str, xAxisLabel: str, yAxisLabel: str, config: List[PlotConfig]):
    fig, ax = plt.subplots()
    # Dictionaries to hold plot data and line objects
    plots = {}
    colors = {}

    # Set plot limits
    ax.set_xlim(0, 100)
    ax.set_ylim(-500, 500)

    # Set labels and title
    ax.set_xlabel(xAxisLabel)
    ax.set_ylabel(yAxisLabel)
    ax.set_title(title)

    for conf in config:
        plots[conf.name] = {'xdata': [], 'ydata': []}
        colors[conf.name] = conf.color
        plots[conf.name]['line'], = ax.plot([], [], color=conf.color, label=conf.name)

    ax.legend()
    plt.ion()

    def update(name: str, new_data: Tuple[float, float]):
        x, y = new_data
        plots[name]['xdata'].append(x)
        plots[name]['ydata'].append(y)
        plots[name]['line'].set_data(plots[name]['xdata'], plots[name]['ydata'])

        ax.relim()
        ax.autoscale_view()

        plt.draw()

    return update