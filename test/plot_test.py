import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pickle

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

wrong_grid_search_1 = load_data("../cup_experiment/grid_search_result/grid_search_1_result.data")

lst2 = [item[0]  for item in wrong_grid_search_1]
param = np.asarray(lst2)
param[:,0]
mean = np.array(wrong_grid_search_1)
mean = mean[:,1]
mean

x = param[:,0],
y = param[:,1],
z = mean
fig = go.Figure(data =
    go.Contour(
        z=z,
        x=x, # horizontal axis
        y=y # vertical axis
    ))
fig.show()