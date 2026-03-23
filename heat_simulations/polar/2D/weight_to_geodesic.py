from curvature_flow import *
import numpy as np
import matplotlib.pyplot as plt

def calculate_geodesic(p, v):
    '''p is initial geodesic position, v is initial velocity'''
    frames, frame_times = sim_in_polar()
    