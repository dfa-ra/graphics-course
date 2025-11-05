import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

from scene_app import SceneApp

root = tk.Tk()
root.title("Освещенность на плоскости — ЛР3 АКГ")

scene_app = SceneApp(root)