import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def positionImpact(data):
    marker_size = 15
    plt.scatter(data[:,0], data[:,1], marker_size, c=data[:,2], cmap='viridis')
    # plt.title("Point observations")
    plt.xlabel("initial acc.")
    plt.ylabel("personalization acc.")
    cbar= plt.colorbar()
    cbar.set_label("weight of each position", labelpad=+1)
    # plt.show()
    plt.savefig('figures/positionImpact.svg', format='svg')


if __name__ == '__main__':
    data = np.random.random((100, 3))
    positionImpact(data)