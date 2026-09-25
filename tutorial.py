#!/usr/bin/env python3
"""
Almost-Linear RNNs Tutorial

This tutorial demonstrates the implementation and training of Almost-Linear RNNs (AL-RNNs)
for modeling dynamical systems, specifically the Lorenz 63 attractor.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import os
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from dataset import TimeSeriesDataset
from linear_region_functions import *
import linear_region_functions as lrf
from metrics import state_space_divergence_binning, power_spectrum_error

from matplotlib.colors import Normalize
from collections import Counter
import seaborn as sns
import copy


class AL_RNN(nn.Module):
    """Almost-Linear RNN Architecture"""

    def __init__(self, M, P, N, relu_mask=None):
        super(AL_RNN, self).__init__()

        # Initialize model dimensions
        self.M = M  # Total number of units
        self.P = P  # Number of piecewise linear units (original ReLU slots)
        self.N = N  # Number of readout units

        # Optional fixed activation mask over the last P slots:
        #   relu_mask[j] = True  → slot j stays ReLU
        #   relu_mask[j] = False → slot j is identity (linearized)
        # Index convention (must match graph reduction):
        #   deleted_relu_local_index j
        #       = j-th element among the last P activation slots
        #       → global hidden index = M - P + j
        # relu_mask=None keeps the original behavior (all last P slots ReLU).
        if relu_mask is None:
            self.relu_mask = None
        else:
            mask = torch.as_tensor(relu_mask, dtype=torch.bool).reshape(-1)
            if mask.numel() != P:
                raise ValueError(
                    f"relu_mask must have length P={P}, got {mask.numel()}")
            # Registered as a (persistent) buffer: saved in state_dict,
            # moves with .to(device), but is NOT a trainable parameter.
            self.register_buffer('relu_mask', mask)

        # Initialize model parameters A, W, h, B
        self.A, self.W, self.h = self.initialize_AWh_random()
        self.B = self.init_uniform((self.N, self.M))

    @property
    def P_effective(self):
        """Number of slots that actually apply ReLU (= P when relu_mask is None)."""
        if self.relu_mask is None:
            return self.P
        return int(self.relu_mask.sum().item())

    def forward(self, z):
        # Make a copy of the input tensor to retain unactivated values
        z_unactivated = torch.clone(z)

        if self.relu_mask is None:
            # Apply ReLU activation on the last P units (original behavior)
            z[:, -self.P:] = F.relu(z[:, -self.P:])
            z_activated = z
        else:
            # Mixed ReLU/identity on the last P slots; no in-place ops on z
            slots = z[:, -self.P:]
            activated_slots = torch.where(self.relu_mask, F.relu(slots), slots)
            z_activated = torch.cat([z[:, :self.M - self.P], activated_slots], dim=-1)

        # Compute the forward pass
        return self.A * z_unactivated + z_activated @ self.W.t() + self.h
    
    def initialize_AWh_random(self):
        """Randomly initialize A, W, h"""
        A = nn.Parameter(torch.diagonal(self.normalized_positive_definite(self.M), 0))
        W = nn.Parameter(torch.randn(self.M, self.M) * 0.01)
        h = nn.Parameter(torch.zeros(self.M))
        return A, W, h
    
    def normalized_positive_definite(self, M):
        """Generate a normalized positive definite matrix"""
        R = np.random.randn(M, M).astype(np.float32)
        K = np.matmul(R.T, R) / M + np.eye(M)  # R'R ./ M + I
        eigenvalues = np.linalg.eigvals(K)
        lambda_max = np.max(np.abs(eigenvalues))
        return torch.tensor(K / lambda_max).float()
    
    def init_uniform(self, shape):
        """Initialize a tensor with a uniform distribution within range [-1/sqrt(M), 1/sqrt(M)]"""
        tensor = torch.empty(*shape)
        r = 1 / math.sqrt(shape[0])
        torch.nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor, requires_grad=True)


@torch.no_grad()
def predict_free_sequence(model, x, T):
    """Predicts a sequence without updating model parameters"""
    b, N = x.size()

    Z = torch.empty(size=(T, b, model.M), device=x.device)
    z = x @ model.B  # Initialize first latent state
    z[:, 0:N] = x

    # Predict sequence by passing previous state through the model
    for t in range(0, T):
        z = model(z)  
        Z[t] = z    
    return Z.permute(1, 0, 2)


def predict_sequence_using_gtf(model, x, alpha, n_interleave):
    """Predicts a sequence using teacher forcing (only for training)"""
    x_ = x.permute(1, 0, 2)  # Permute input to shape (sequence_length, batch_size, feature_dim)
    T, b, dx = x_.size()  # T: sequence length, b: batch size, dx: feature dimension
    Z = torch.empty(size=(T, b, model.M), device=x.device)
    z = x_[0] @ model.B  # Initialize first latent state
    z = teacher_force(z, x_[0], alpha=1)  # Apply teacher forcing to the initial state

    # Generate sequence predictions
    for t in range(0, T):
        # Apply teacher forcing at regular intervals
        if (t % n_interleave == 0) and (t > 0):
            z = teacher_force(z, x_[t], alpha)
            
        # Update the latent state using the model
        z = model(z)
        Z[t] = z
    return Z.permute(1, 0, 2)


def teacher_force(z, x, alpha):
    """Teacher force the state z"""
    N = x.shape[1]  # Get N from the input dimensions
    z[:, :N] = alpha * x + (1 - alpha) * z[:, :N]
    return z


def train_sh(model, dataset, optimizer, scheduler, loss_fn, num_epochs, alpha, n_interleave, 
             batches_per_epoch=50, ssi=25, use_best_model=True):
    """Training routine with scheduled hyperparameter updates"""
    model.train()
    best_model = copy.deepcopy(model)
    losses = []
    klx = []
    dh = []

    # create the save directory if it does not exist
    os.makedirs("models", exist_ok=True)
    
    with trange(num_epochs, desc="Training Progress") as epochs:
        for e in epochs:
            epoch_losses = []
            
            for _ in range(batches_per_epoch):
                optimizer.zero_grad()
                
                x, y, s = dataset.sample_batch()
                z_hat = predict_sequence_using_gtf(model, x, alpha, n_interleave)
                loss = loss_fn(z_hat[:, :, :model.N], y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            scheduler.step()
            
            # Compute and store average loss for the epoch
            average_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            epochs.set_postfix(loss=average_epoch_loss)
            losses.append(average_epoch_loss)

            # Save dynamical systems performance metrics Dstsp and DH each ssi epochs
            if e % ssi == 0:
                with torch.no_grad():
                    z_test = predict_free_sequence(model, dataset.X.clone().detach()[0:1, :], 10000)
                    klx.append(state_space_divergence_binning(z_test[0, :, 0:model.N], dataset.X.clone().detach()))
                    dh.append(power_spectrum_error(z_test[0, :, 0:model.N], dataset.X.clone().detach()[0:10000, :]))
                    
                    if torch.argmin(torch.tensor(klx)) + 1 == len(torch.tensor(klx)):
                        best_model = copy.deepcopy(model)

            # save the model every 500 epochs
            # loop variable e starts at 0, so the epoch count is (e + 1)
            epochs_for_save = 500
            if (e + 1) % epochs_for_save == 0:
                # build the save path including the epoch count
                # save_path = f"models/lorenz_m{model.M}_p{model.P}_epoch{e + 1}of{num_epochs}.pth"
                save_path = f"models/chua_3scroll_m{model.M}_p{model.P}_epoch{e + 1}of{num_epochs}.pth"
                torch.save(best_model.state_dict(), save_path) # save the best model so far
                # show a message next to the tqdm progress bar
                epochs.set_description(f"Training Progress (Saved best @ epoch {e + 1})")

    if use_best_model:
        model.load_state_dict(best_model.state_dict())
    
    return [losses, klx, dh]


def main():
    """Main function to run the AL-RNN tutorial"""
    
    # Load Data
    print("Loading data...")
    # X_train = np.load("lorenz63_train.npy").astype(np.float32)[500:]
    # X_test = np.load("lorenz63_test.npy").astype(np.float32)[500:]
    X_train = np.load("chua_3-scroll_train.npy").astype(np.float32)
    X_test = np.load("chua_3-scroll_test.npy").astype(np.float32)
    T_train, N = X_train.shape
    T_test = X_test.shape[0]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

    # Initialize Model
    print("\nInitializing model...")
    N = X_train.shape[-1]  # number readout units
    M = 20  # number of units in total 3.20s/it
    # M = 30
    # M = 10 # 2.05s/it
    # P = 2   # number of piecewise linear units
    P = 6  # number of piecewise linear units

    model = AL_RNN(M=M, P=P, N=N)
    print(f"Model initialized with M={M}, P={P}, N={N}")

    # Training hyperparameters
    batch_size = 16
    sequence_length = 200
    alpha = 1  # teacher forcing alpha (GTF)
    n_interleave = 16  # teacher forcing interval

    loss_fn = nn.MSELoss()
    dataset = TimeSeriesDataset(X_train, sequence_length=sequence_length, batch_size=batch_size)

    # Optimization
    num_epochs = 2000 # original
    # num_epochs = 10000
    start_learning_rate = 1e-3
    optimizer = torch.optim.RAdam(model.parameters(), lr=start_learning_rate)
    end_learning_rate = 1e-5
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=np.exp(np.log(end_learning_rate/start_learning_rate)/num_epochs)
    )

    # Training
    print("\nStarting training...")
    metrics = train_sh(model, dataset, optimizer, scheduler, loss_fn, 
                       num_epochs=num_epochs, alpha=alpha, n_interleave=n_interleave, 
                       batches_per_epoch=50, ssi=20)

    # Plot training loss
    plt.rcParams["figure.figsize"] = (7, 4)
    plt.rcParams.update({'font.size': 10})
    fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.plot(metrics[0], lw=3)
    ax.set_title('Training loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(f'figures/loss_m{M}_p{P}_{num_epochs}epochs.png')
    plt.close()

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Save model
    print("\nSaving model...")
    # torch.save(model.state_dict(), f"models/lorenz_m{M}_p{P}_{num_epochs}epochs")
    torch.save(model.state_dict(), f"models/chua_3scroll_m{M}_p{P}_{num_epochs}epochs")

    # Load model (demonstration)
    model = AL_RNN(M=M, P=P, N=N)
    # model.load_state_dict(torch.load(f"models/lorenz_m{M}_p{P}_{num_epochs}epochs"))
    model.load_state_dict(torch.load(f"models/chua_3scroll_m{M}_p{P}_{num_epochs}epochs"))

    # Analysis
    print("\nRunning analysis...")
    X_test_torch = torch.tensor(X_test[:]).unsqueeze(0)

    T_gen = 10000  # Sequence length
    T_r = 1000     # Transient cutoff length
    orbit = predict_free_sequence(model, X_test_torch[:, 0, :], T_gen + T_r).detach().numpy()[0][T_r:, :]

    Dstsp = state_space_divergence_binning(torch.tensor(orbit[:, 0:model.N]), X_test_torch[0, :, :])
    DH = power_spectrum_error(torch.tensor(orbit[:, 0:model.N]), X_test_torch[0, 0:T_gen, :])
    print(f"State space distance (Dstsp): {Dstsp}")
    print(f"Hellinger Distance (DH): {DH}")

    # Plot attractor comparison
    Blues = plt.cm.Blues
    plt.rcParams["lines.linewidth"] = .35
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams["lines.linewidth"] = 2.
    plt.rcParams.update({'font.size': 10})
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = orbit[:, 0]
    ys = orbit[:, 1]
    zs = orbit[:, 2]

    ax.plot(X_test[:T_gen, 0], X_test[:T_gen, 1], X_test[:T_gen, 2], 
            color=Blues(0.9), label="Ground Truth")
    ax.plot(xs, ys, zs, color=Blues(0.6), alpha=1., label="Freely Generated")

    plt.title(r'$D_{stsp}={Dstsp}, D_H={DH}$')

    plt.legend(loc="upper left")
    plt.axis("off")
    plt.savefig(f'figures/trajectories_m{M}_p{P}_{num_epochs}epochs.png')
    plt.close()

    # Linear region analysis
    print("\nAnalyzing linear regions...")
    generated_latent = orbit[:, -P:]  # latent sequence in PWL units
    generated_observations = orbit[:, :N]  # predicted readout sequence 

    bits = lrf.convert_to_bits(generated_latent)
    regions, unique_regions = lrf.unique_regions_crossed(bits, M)
    frequencies = lrf.frequency_of_regions(bits, unique_regions)

    # Compute order of most frequent visited subregions
    frequency_list = np.copy(frequencies)
    regions_list = np.copy(unique_regions)    
    most_frequent_regions = []
    for _ in range(len(frequency_list)):
        index = np.argmax(frequency_list)
        most_frequent_regions.append(regions_list[index])
        frequency_list = np.delete(frequency_list, index)
        regions_list = np.delete(regions_list, index, axis=0)

    connectome = lrf.connectome_with_self_connections(bits, most_frequent_regions, len(most_frequent_regions))
    print(f"Used subregions: {len(connectome)}")
    print(f"Timesteps in subregion: {frequencies}")

    # Plot connectome heatmap
    plt.figure(figsize=(7, 5))
    sns.set_context('talk', font_scale=1.2) 
    sns.set_style('white')

    cmap = plt.get_cmap('Blues')
    ax = sns.heatmap(data=connectome[:, :], annot=False, linewidths=0, cmap=cmap, square=True,
                     yticklabels=False, xticklabels=False, cbar_kws={'label': 'Transition Probability'})

    cbar = ax.collections[0].colorbar
    cbar.set_label('Transition Probability', rotation=270, labelpad=35)
    ax.collections[0].set_clim(0, np.max(connectome)-0.015)

    plt.savefig(f'figures/transition probability_m{M}_p{P}_{num_epochs}epochs.png')
    plt.close()

    # Plot colored trajectory analysis
    sns.set_context('talk', font_scale=1.0) 
    plt.rcParams["lines.linewidth"] = 2.

    observations_plot = generated_observations[:, :3]
    bitcode_plot = lrf.convert_to_bits(generated_latent)

    bitcodes_str = [''.join(map(str, map(int, b))) for b in bitcode_plot]
    bitcode_freq = Counter(bitcodes_str)
    most_frequent_regions = [item[0] for item in bitcode_freq.most_common()]
    frequency_order_map = {code: idx for idx, code in enumerate(most_frequent_regions)}
    order = [frequency_order_map[code] for code in bitcodes_str]

    order_array = np.array(order)
    norm = Normalize(vmin=order_array.min(), vmax=order_array.max())
    cmap = plt.get_cmap('cividis')
    colors = cmap(norm(order_array))

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(121, projection='3d')
    scatter = ax.scatter(observations_plot[:, 0], observations_plot[:, 1], observations_plot[:, 2], 
                        c=colors, s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis("off")

    ax = fig.add_subplot(322)
    for i in range(1000):
        ax.plot([i, i+1], [observations_plot[i, 0], observations_plot[i+1, 0]], color=colors[i+1])
    ax.set_ylabel('X')

    ax = fig.add_subplot(324)
    for i in range(1000):
        ax.plot([i, i+1], [observations_plot[i, 1], observations_plot[i+1, 1]], color=colors[i+1])
    ax.set_ylabel('Y')

    ax = fig.add_subplot(326)
    for i in range(1000):
        ax.plot([i, i+1], [observations_plot[i, 2], observations_plot[i+1, 2]], color=colors[i+1])
    ax.set_ylabel('Z')

    plt.tight_layout()
    plt.savefig(f'figures/linear_subreginos_m{M}_p{P}_{num_epochs}epochs.png')
    plt.close()

    print("\nTutorial completed!")


if __name__ == "__main__":
    main()