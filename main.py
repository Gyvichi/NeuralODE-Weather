```
rk4 solver, AdamW optimizer, 

```
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Constants from Delhi module
features = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
units = ["Celcius", "g/m³ of water", "km/h", "hPa"]
feature_names = ["Mean temperature", "Humidity", "Wind speed", "Mean pressure"]

### Delhi Data Handling Functions
def load():
    """Loads the Delhi dataset by concatenating train and test CSV files."""
    df_train = pd.read_csv("data/DailyDelhiClimateTrain.csv")
    df_test = pd.read_csv("data/DailyDelhiClimateTest.csv")
    df = pd.concat([df_train, df_test], ignore_index=True)
    return df

def plot_features(df):
    """Plots each feature as a time series."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for i, (f, n, u) in enumerate(zip(features, feature_names, units)):
        ax = axes[i // 2, i % 2]
        ax.plot(df['date'], df[f], label=n, color=f'C{i}')
        ax.set_title(n)
        ax.set_ylabel(u)
        ax.set_xlabel('Time')
        ax.legend()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/features.svg")
    plt.close()

def normalize(x):
    """Normalizes data by subtracting mean and dividing by standard deviation."""
    mu = np.mean(x, axis=1, keepdims=True)
    sigma = np.std(x, axis=1, keepdims=True)
    z = (x - mu) / sigma
    return z, mu, sigma

def preprocess(raw_df, num_train=20):
    """Preprocesses the Delhi dataset by grouping and normalizing."""
    raw_df['date'] = pd.to_datetime(raw_df['date'])
    raw_df['year'] = raw_df['date'].dt.year
    raw_df['month'] = raw_df['date'].dt.month
    df = raw_df.groupby(['year', 'month']).agg({
        'date': lambda d: d.mean().year + d.mean().month / 12,
        'meantemp': 'mean',
        'humidity': 'mean',
        'wind_speed': 'mean',
        'meanpressure': 'mean'
    }).reset_index(drop=True)

    t_and_y = lambda df: (df['date'].values, df[features].values.T)
    t_train, y_train = t_and_y(df.iloc[:num_train])
    t_test, y_test = t_and_y(df.iloc[num_train:])

    t_train, t_mean, t_scale = normalize(t_train.reshape(1, -1))
    y_train, y_mean, y_scale = normalize(y_train)
    t_test = (t_test - t_mean) / t_scale
    y_test = (y_test - y_mean) / y_scale

    return (
        t_train.flatten(), y_train,
        t_test.flatten(), y_test,
        (t_mean.item(), t_scale.item()),
        (y_mean.flatten(), y_scale.flatten())
    )

### NeuralODE Model Definition
class ODEFunc(nn.Module):
    """Defines the neural network for the ODE function."""
    def __init__(self, data_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 64),
            nn.SiLU(),  # Swish activation
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, data_dim)
        )

    def forward(self, t, y):
        return self.net(y)

### Training Functions
def train_one_round(func, y0, t, y, opt, maxiters):
    """Trains the model for one round and returns losses."""
    losses = []
    for _ in range(maxiters):
        opt.zero_grad()
        y_pred = odeint(func, y0, t, method='rk4', atol=1e-9, rtol=1e-9)
        loss = torch.sum((y_pred - y) ** 2)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses

def train(t, y, obs_grid, maxiters, lr):
    """Trains the NeuralODE model over multiple rounds, logging losses and states."""
    data_dim = y.shape[0]
    func = ODEFunc(data_dim)
    opt = optim.AdamW(func.parameters(), lr=lr)

    total_losses = []
    state_dicts = []
    torch.manual_seed(123)  # For reproducibility
    for _ in range(len(obs_grid)):
        losses = train_one_round(func, y[:, 0], t, y, opt, maxiters)
        total_losses.extend(losses)
        state_dicts.append(func.state_dict().copy())  # Save model state
    return func, total_losses, state_dicts

### Prediction and Rescaling Functions
def predict(func, y0, t):
    """Predicts the trajectory using the trained model."""
    with torch.no_grad():
        return odeint(func, y0, t, method='rk4', atol=1e-9, rtol=1e-9)

def rescale_t(t, t_mean, t_scale):
    """Rescales normalized time to original scale."""
    return t * t_scale + t_mean

def rescale_y(y, y_mean, y_scale):
    """Rescales normalized features to original scale."""
    return y * y_scale + y_mean

### Plotting and Animation Functions
def plot_result(t_train, y_train, t_grid, y_pred, loss, num_iters):
    """Plots predictions and loss for all features."""
    fig, axes = plt.subplots(5, 1, figsize=(9, 9), gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]})
    for i, ax in enumerate(axes[:-1]):
        ax.scatter(t_train, y_train[i], label='Observation', color=f'C{i}', marker='o')
        ax.plot(t_grid, y_pred[i], label='Prediction', color=f'C{i}', linewidth=2)
        ax.set_title(feature_names[i])
        ax.set_xlabel('Time')
        ax.set_ylabel(units[i])
        ax.legend()
        if i == 0:
            ax.set_ylim(10, 40)
        elif i == 1:
            ax.set_ylim(20, 100)
        elif i == 2:
            ax.set_ylim(2, 12)
        elif i == 3:
            ax.set_ylim(990, 1025)
    axes[-1].plot(loss, label='Loss', linewidth=2)
    axes[-1].set_title('Loss')
    axes[-1].set_xlabel('Iterations')
    axes[-1].set_ylabel('Loss')
    axes[-1].set_xlim(0, num_iters)
    axes[-1].legend()
    plt.tight_layout()
    return fig

def animate_training(t_train, y_train, t_grid, func, state_dicts, losses, obs_grid):
    """Creates an animation of the training process."""
    obs_count = {i: n for i, n in enumerate(obs_grid)}
    total_frames = len(losses) + 300  # Pause frames
    frame_indices = list(range(2, total_frames + 1, 2))  # Every 2 frames

    fig = plt.figure(figsize=(9, 9))
    axes = fig.add_subplots([5, 1], gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]})

    def update(i):
        for ax in axes:
            ax.clear()
        stage = min(int((i - 1) / len(losses) * len(obs_grid)), len(obs_grid) - 1)
        k = obs_count[stage]

        # Load model state for this round
        func.load_state_dict(state_dicts[stage])
        y_pred = predict(func, y_train[:, 0], t_grid).detach().numpy().T

        for idx, ax in enumerate(axes[:-1]):
            ax.scatter(t_train[:k], y_train[idx, :k], label='Observation', color=f'C{idx}')
            ax.plot(t_grid.numpy(), y_pred[idx], label='Prediction', color=f'C{idx}', linewidth=2)
            ax.set_title(feature_names[idx])
            ax.set_xlabel('Time')
            ax.set_ylabel(units[idx])
            ax.legend()
            if idx == 0:
                ax.set_ylim(10, 40)
            elif idx == 1:
                ax.set_ylim(20, 100)
            elif idx == 2:
                ax.set_ylim(2, 12)
            elif idx == 3:
                ax.set_ylim(990, 1025)
        axes[-1].plot(losses[:i], label='Loss', linewidth=2)
        axes[-1].set_title('Loss')
        axes[-1].set_xlabel('Iterations')
        axes[-1].set_ylabel('Loss')
        axes[-1].set_xlim(0, len(losses))
        axes[-1].legend()
        plt.tight_layout()

    anim = FuncAnimation(fig, update, frames=frame_indices, interval=33, repeat=False)  # ~30 fps
    anim.save("plots/training.gif", writer='pillow')
    plt.close()

def plot_extrapolation(t_train, y_train, t_test, y_test, t_grid, y_pred):
    """Plots training data, test data, and predictions."""
    fig, axes = plt.subplots(4, 1, figsize=(9, 9))
    for i, ax in enumerate(axes):
        ax.scatter(t_train, y_train[i], label='Train', color=f'C{i}', marker='o')
        ax.scatter(t_test, y_test[i], label='Test', color=f'C{i}', marker='x')
        ax.plot(t_grid, y_pred[i], label='Prediction', color=f'C{i}', linewidth=2)
        ax.set_title(feature_names[i])
        ax.set_xlabel('Time')
        ax.set_ylabel(units[i])
        ax.legend()
        if i == 0:
            ax.set_ylim(10, 40)
        elif i == 1:
            ax.set_ylim(20, 100)
        elif i == 2:
            ax.set_ylim(2, 12)
        elif i == 3:
            ax.set_ylim(990, 1025)
    plt.tight_layout()
    plt.savefig("plots/extrapolation.svg")
    plt.close()

### Main Execution
if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load()
    plot_features(df)
    t_train, y_train, t_test, y_test, (t_mean, t_scale), (y_mean, y_scale) = preprocess(df, num_train=20)

    # Convert to PyTorch tensors
    t_train = torch.tensor(t_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    t_test = torch.tensor(t_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # Training parameters
    obs_grid = list(range(4, len(t_train) + 1, 4))
    maxiters = 150
    lr = 5e-3

    # Train the model
    print("Fitting model...")
    func, losses, state_dicts = train(t_train, y_train, obs_grid, maxiters, lr)

    # Define fine time grid for predictions
    t_grid = torch.linspace(t_train.min(), t_test.max(), 500)

    # Generate training animation
    print("Generating training animation...")
    animate_training(t_train, y_train, t_grid, func, state_dicts, losses, obs_grid)

    # Generate extrapolation plot
    print("Generating extrapolation plot...")
    y_pred = predict(func, y_train[:, 0], t_grid).detach().numpy().T
    t_train_rescaled = rescale_t(t_train.numpy(), t_mean, t_scale)
    y_train_rescaled = rescale_y(y_train.numpy(), y_mean, y_scale)
    t_test_rescaled = rescale_t(t_test.numpy(), t_mean, t_scale)
    y_test_rescaled = rescale_y(y_test.numpy(), y_mean, y_scale)
    t_grid_rescaled = rescale_t(t_grid.numpy(), t_mean, t_scale)
    y_pred_rescaled = rescale_y(y_pred, y_mean, y_scale)
    plot_extrapolation(t_train_rescaled, y_train_rescaled, t_test_rescaled, y_test_rescaled, t_grid_rescaled, y_pred_rescaled)

    print("Done!")
