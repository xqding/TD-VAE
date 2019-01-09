import numpy as np
import matplotlib.pyplot as plt

loss = []
with open('./log/loginfo.txt', 'r') as file_handle:
    for line in file_handle:
        line = line.strip()
        field = line.split()
        if field[-1] != "nan":
            loss.append(float(field[-1]))
            
plt.plot(loss[::50])
plt.ylim(10,300)
plt.show()
