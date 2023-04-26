import re

import matplotlib.pyplot as plt

import numpy as np




# Load the data from the text file

data = np.loadtxt('fromevo.txt', dtype="str")

data = np.array([

 complex(line.replace('i', 'j'))
 for line in data

])




# Get the real and imaginary parts of the data

real_data = data.real

imag_data = data.imag




# Create a time array

time = np.arange(len(data))




# Create a figure with two subplots

fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 6))




# Plot the real and imaginary parts of the signal

ax.plot(time, real_data, time, imag_data)




# Add labels to the subplots

ax.set_xlabel("Time")




# Add a title to the figure

ax.set_title("Complex Signal")




# Display the plot

plt.show()