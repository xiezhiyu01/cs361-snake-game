import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
import os

class SnakeRenderer:
    def __init__(self, grid_size, save_dir=None):
        self.grid_size = grid_size
        self.fig = None
        self.ax = None
        self.save_dir = save_dir # or None if you don't want to save frames
        self.frame_count = 0

    def render(self, state):
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            plt.subplots_adjust(bottom=0.2)

        self.ax.clear()
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_aspect('equal')

        direction = state['direction']
        for i, (x, y) in enumerate(state['snake']):
            if i == 0:
                if direction == (-1, 0):
                    triangle = patches.Polygon([(y, x + 1), (y + 1, x + 1), (y + 0.5, x)], color='darkgreen')
                elif direction == (1, 0):
                    triangle = patches.Polygon([(y, x), (y + 1, x), (y + 0.5, x + 1)], color='darkgreen')
                elif direction == (0, -1):
                    triangle = patches.Polygon([(y + 1, x), (y + 1, x + 1), (y, x + 0.5)], color='darkgreen')
                elif direction == (0, 1):
                    triangle = patches.Polygon([(y, x), (y, x + 1), (y + 1, x + 0.5)], color='darkgreen')
                else:
                    triangle = patches.RegularPolygon((y + 0.5, x + 0.5), 3, radius=0.35, color='darkgreen')
                self.ax.add_patch(triangle)
            else:
                self.ax.add_patch(patches.Rectangle((y, x), 1, 1, facecolor='lightgreen', edgecolor='green'))

        fx, fy = state['food']
        self.ax.add_patch(patches.Circle((fy + 0.5, fx + 0.5), radius=0.4, color='red'))
        self.ax.set_title("Snake")
        if self.save_dir:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            filepath = f"{self.save_dir}/frame_{self.frame_count:04d}.png"
            self.fig.savefig(filepath, dpi=100)
            self.frame_count += 1
        else:
            plt.draw()
            plt.pause(0.01)
