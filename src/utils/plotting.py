import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation
import io
from PIL import Image
import torch

def plot_losses(losses):
    # Handle empty or invalid data
    if not losses or not all(len(losses[key]) > 0 for key in losses):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No loss data available", fontsize=12, ha='center', va='center')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return buf

    # Convert lists to numpy arrays for easier handling
    loss_types = ['total', 'pde', 'bc', 'ic', 'data']
    max_len = max(len(losses[key]) for key in loss_types if len(losses[key]) > 0)
    epochs = np.arange(max_len)

    fig, ax = plt.subplots(figsize=(10, 6))
    for key in loss_types:
        if len(losses[key]) > 0:
            # Pad shorter lists with NaN to match max length
            data = np.array(losses[key])
            if len(data) < max_len:
                data = np.pad(data, (0, max_len - len(data)), mode='constant', constant_values=np.nan)
            # Plot with log scale for better visibility
            ax.plot(epochs, data, label=key.capitalize() + ' Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (Log Scale)')
    ax.set_title('Training Losses')
    ax.set_yscale('log')  # Use logarithmic scale
    ax.legend()
    ax.grid(True)
    
    # Set reasonable y-axis limits to avoid empty plots
    ax.set_ylim(custom_ylim(losses))
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def custom_ylim(losses):
    # Compute reasonable y-axis limits based on non-NaN, non-inf values
    all_losses = []
    for key in losses:
        if len(losses[key]) > 0:
            data = np.array(losses[key])
            data = data[np.isfinite(data)]  # Exclude NaN and inf
            all_losses.extend(data)
    if not all_losses:
        return 1e-3, 1e3  # Default range if no valid data
    min_val = max(min(all_losses), 1e-10)  # Avoid zero or negative for log scale
    max_val = max(all_losses)
    return min_val * 0.5, max_val * 2

def plot_reservoir_data(data_df):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(data_df['x'], data_df['y'], data_df['z'], c=data_df['p'], cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.colorbar(scatter, ax=ax, label='Pressure (Pa)')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def plot_pressure_slice(model, t_val, data_df, device, z_slice=0.5, p_scale=1000.0, p_base=1000.0):
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_slice)
    T = np.full_like(X, t_val)
    
    inputs = torch.tensor(np.stack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()], axis=1), 
                         dtype=torch.float32).to(device)
    with torch.no_grad():
        P = model(inputs, p_scale, p_base).cpu().numpy().reshape(X.shape)
    
    fig = plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, P, levels=20, cmap="viridis")
    plt.colorbar(label="Pressure (Pa)")
    plt.scatter(data_df['x'], data_df['y'], c="red", label="Data points (projected)")
    plt.title(f"Pressure Slice at z={z_slice}, t={t_val:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

def animate_pressure(model, data_df, device, z_slice=0.5, p_scale=1000.0, p_base=1000.0, save_format='gif'):
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_slice)
    times = np.linspace(0, 1, 50)

    fig, ax = plt.subplots(figsize=(8, 6))
    inputs = torch.tensor(np.stack([X.flatten(), Y.flatten(), Z.flatten(), np.zeros_like(X.flatten())], axis=1), 
                         dtype=torch.float32).to(device)

    def update(frame):
        ax.clear()
        t_val = times[frame]
        inputs[:, 3] = t_val
        with torch.no_grad():
            P = model(inputs, p_scale, p_base).cpu().numpy().reshape(X.shape)
        contour = ax.contourf(X, Y, P, levels=20, cmap="viridis")
        mask = np.abs(data_df['z'] - z_slice) < 0.1
        ax.scatter(data_df['x'][mask], data_df['y'][mask], c="red", label="Data points")
        ax.set_title(f"Pressure at z={z_slice:.2f}, t={t_val:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        if frame == 0:
            fig.colorbar(contour, ax=ax, label="Pressure (Pa)")
        return contour,

    anim = FuncAnimation(fig, update, frames=len(times), interval=100, blit=False)
    filename = 'pressure_animation.gif'
    anim.save(filename, writer='pillow', fps=10)
    plt.close(fig)
    return filename