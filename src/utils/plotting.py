import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation
import io
from PIL import Image
import torch

def plot_losses(losses):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses['total'], label='Total Loss')
    ax.plot(losses['pde'], label='PDE Loss')
    ax.plot(losses['bc'], label='BC Loss')
    ax.plot(losses['ic'], label='IC Loss')
    ax.plot(losses['data'], label='Data Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

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
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_slice)
    times = np.linspace(0, 1, 50)  # 50 frames from t=0 to t=1

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
        ax.scatter(data_df['x'], data_df['y'], c="red", label="Data points (projected)")
        ax.set_title(f"Pressure Slice at z={z_slice}, t={t_val:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        if frame == 0:
            fig.colorbar(contour, ax=ax, label="Pressure (Pa)")
        return contour,

    anim = FuncAnimation(fig, update, frames=len(times), interval=100, blit=False)
    filename = 'pressure_animation.gif' if save_format == 'gif' else 'pressure_animation.mp4'
    if save_format == 'gif':
        anim.save(filename, writer='pillow', fps=10)
    else:
        anim.save(filename, writer='ffmpeg', fps=10)
    plt.close(fig)
    return filename