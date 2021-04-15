import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px


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


def plot_reconstructions(model, train_dict, n=5, figsize=15):
    scale = 1.0
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    plt.figure(figsize=(figsize, figsize))
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            signal = train_dict['Torso_Acc_x'][np.random.randint(0, 1000), :]
            _, _, z_sample = model.encode(np.expand_dims(signal, (0,-1)))
            x_decoded = model.decode(z_sample)
            #print(x_decoded)
            plt.subplot(n, n, j+(i*len(grid_x))+1)
            plt.plot(x_decoded[0, :], label='decoded')
            #plt.plot(signal, label='original signal')
            #plt.plot(signal, 'r')
            plt.legend()

    plt.show()


def plot_losses(history):
    fig, ax1 = plt.subplots()

    # b+ is for "blue cross"
    color='tab:red'
    ax1.plot(history.history['loss'], 'b', label='joint loss')
    ax1.plot(history.history['reconstruction_loss'], 'r*', label='reconstruction loss')
    ax1.set_ylabel('Reconstruction loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color='tab:blue'
    ax2.plot(history.history['kl_loss'], 'g+', label='KL divergence')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylabel('KL divergence', color=color)
    plt.xlabel('Epochs')
    plt.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def plot_embedding(model, train_dict, train_labels):
    codes = model.encode(train_dict['Torso_Acc_x'])
    X_embedded = TSNE(n_components=3).fit_transform(codes[2])
    fig = px.scatter_3d(
        X_embedded, x=0, y=1, z=2,
        color=train_labels,  # labels={'color': 'species'}
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig.update_traces(marker_size=8)
    fig.show()


def plot_latent_magnitudes(model, train_dict, train_labels, label):
    fig = plt.figure()
    codes = model.encode(train_dict['Torso_Acc_x'][np.where(train_labels==label)])
    heights = np.absolute(codes[2].numpy().mean(axis=0))
    plt.bar(x=np.arange(1, codes[2].shape[1]+1), height=heights, align="center")
    plt.xlabel('Latent dimension')
    plt.ylabel('Avg. latent magnitude')
    plt.savefig('average-latent-magnitude.svg', format='svg')
    fig.show()
