# import library
import matplotlib.pyplot as plt
import numpy as np
from arlpy import unet 
s = unet.get_signals("fromevo.txt")

with open("fromevo.txt") as f:
    lines = f.readlines()
    x = [line.split()[0] for line in lines]


complex_arr = [complex(s.replace("i", "j")) for s in x]


print(complex_arr)


time = np.arange(len(data))




# Create a figure with two subplots

fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 6))




# Plot the real and imaginary parts of the signal

ax.plot(time, real_data, time, imag_data)