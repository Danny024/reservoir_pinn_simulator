import streamlit as st
import torch
import pandas as pd
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import os
import requests
import mlflow
import mlflow.pytorch
from src.models.pinn import PINN
from src.utils.training import compute_pde_residual
from src.utils.plotting import plot_losses, plot_reservoir_data, plot_pressure_slice, animate_pressure
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY", "")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up MLflow experiment
mlflow.set_experiment("3D_Reservoir_PINN")

# Streamlit app
st.title("3D Reservoir PINN Simulator")

# Sidebar for data visualization and training updates
with st.sidebar:
    st.header("3D Reservoir Data Visualization")
    if 'data_df' in st.session_state and st.session_state.data_df is not None:
        buf = plot_reservoir_data(st.session_state.data_df)
        st.image(buf, caption="3D Scatter Plot of Reservoir Data")
    else:
        st.info("Upload a CSV file with x, y, z, t, p columns to visualize data.")

    st.header("Training Progress")
    progress_placeholder = st.empty()
    epoch_placeholder = st.empty()
    loss_placeholder = st.empty()
    run_id_placeholder = st.empty()

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = PINN().to(device)
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'phi' not in st.session_state:
    st.session_state.phi = 0.2
if 'k' not in st.session_state:
    st.session_state.k = 0.1
if 'mu' not in st.session_state:
    st.session_state.mu = 1.0
if 'ct' not in st.session_state:
    st.session_state.ct = 1e-6
if 'q0' not in st.session_state:
    st.session_state.q0 = 0.01
if 'well_pos' not in st.session_state:
    st.session_state.well_pos = (0.5, 0.5, 0.5)
if 'p_base' not in st.session_state:
    st.session_state.p_base = 1000.0
if 'p_scale' not in st.session_state:
    st.session_state.p_scale = 1000.0
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = "DeepSeek" if not openai_api_key else "OpenAI"
if 'losses' not in st.session_state:
    st.session_state.losses = {'pde': [], 'bc': [], 'ic': [], 'data': [], 'total': []}

# Input reservoir parameters
st.header("Reservoir Parameters")
col1, col2 = st.columns(2)
with col1:
    st.session_state.phi = st.number_input("Porosity (phi)", value=0.2, step=0.01)
    st.session_state.k = st.number_input("Permeability (k, Darcy)", value=0.1, step=0.01)
    st.session_state.mu = st.number_input("Viscosity (mu, cP)", value=1.0, step=0.1)
    st.session_state.p_base = st.number_input("Base Pressure (p_base, Pa)", value=1000.0, step=100.0)
with col2:
    st.session_state.ct = st.number_input("Compressibility (ct, 1/Pa)", value=1e-6, step=1e-7, format="%.1e")
    st.session_state.q0 = st.number_input("Injection Rate (q0, m^3/s)", value=0.01, step=0.001)
    st.session_state.p_scale = st.number_input("Pressure Scale (p_scale)", value=1000.0, step=100.0)
    well_x = st.number_input("Well X Position", value=0.5, step=0.1, min_value=0.0, max_value=1.0)
    well_y = st.number_input("Well Y Position", value=0.5, step=0.1, min_value=0.0, max_value=1.0)
    well_z = st.number_input("Well Z Position", value=0.5, step=0.1, min_value=0.0, max_value=1.0)
    st.session_state.well_pos = (well_x, well_y, well_z)

# Upload CSV file
st.header("Upload 3D Reservoir Data")
uploaded_file = st.file_uploader("Choose a CSV file with x, y, z, t, p columns", type="csv")
if uploaded_file:
    data_df = pd.read_csv(uploaded_file)
    if all(col in data_df.columns for col in ['x', 'y', 'z', 't', 'p']):
        st.session_state.data_df = data_df
        st.success("CSV file loaded successfully!")
    else:
        st.error("CSV file must contain columns: x, y, z, t, p")
        st.session_state.data_df = None
else:
    # Try loading default dataset
    try:
        data_df = pd.read_csv("data/reservoir_data.csv")
        if all(col in data_df.columns for col in ['x', 'y', 'z', 't', 'p']):
            st.session_state.data_df = data_df
            st.info("Loaded default dataset: data/reservoir_data.csv")
        else:
            st.session_state.data_df = None
    except FileNotFoundError:
        st.session_state.data_df = None

# Training Parameters and Model Loading
st.header("Training Parameters")
epochs = st.number_input("Number of Epochs", value=20000, step=1000, min_value=1000)

# Load Trained Model from MLflow
st.subheader("Load Trained Model")
try:
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("3D_Reservoir_PINN")
    if experiment:
        runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"])
        run_options = {f"Run ID: {run.info.run_id} (Started: {run.info.start_time})": run.info.run_id for run in runs}
        selected_run_id = st.selectbox("Select MLflow Run", list(run_options.keys()), index=0)
        if st.button("Load Trained Model"):
            with st.spinner("Loading model from MLflow..."):
                model_uri = f"runs:/{run_options[selected_run_id]}/model"
                st.session_state.model = mlflow.pytorch.load_model(model_uri).to(device)
                st.session_state.trained = True
                st.success(f"Loaded model from Run ID: {run_options[selected_run_id]}")
    else:
        st.warning("No MLflow runs found for 3D_Reservoir_PINN experiment.")
except Exception as e:
    st.warning(f"Error accessing MLflow runs: {e}")

# Custom training function with MLflow logging and GUI updates
def train_model_with_updates(model, x_r, y_r, z_r, t_r, x_b, y_b, z_b, t_b, x_i, y_i, z_i, t_i, x_d, y_d, z_d, t_d, p_d, epochs, phi, k, mu, ct, q0, well_pos, p_scale, p_base):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=2000, gamma=0.5)
    scaler = GradScaler()  # For mixed precision training
    losses = {'pde': [], 'bc': [], 'ic': [], 'data': [], 'total': []}
    
    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("phi", phi)
        mlflow.log_param("k", k)
        mlflow.log_param("mu", mu)
        mlflow.log_param("ct", ct)
        mlflow.log_param("q0", q0)
        mlflow.log_param("well_pos_x", well_pos[0])
        mlflow.log_param("well_pos_y", well_pos[1])
        mlflow.log_param("well_pos_z", well_pos[2])
        mlflow.log_param("p_scale", p_scale)
        mlflow.log_param("p_base", p_base)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("N_r", len(x_r))
        mlflow.log_param("N_b", len(x_b))
        mlflow.log_param("N_i", len(x_i))
        mlflow.log_param("N_d", len(x_d))
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Mixed precision training
            with autocast():
                # PDE residual loss
                residual = compute_pde_residual(model, x_r, y_r, z_r, t_r, phi, k, mu, ct, q0, well_pos, p_scale, p_base)
                loss_pde = torch.mean(residual**2)
                
                # Boundary condition loss
                inputs_b = torch.stack([x_b, y_b, z_b, t_b], dim=1).requires_grad_(True).to(x_r.device)
                p_b = model(inputs_b, p_scale, p_base)
                p_grad_b = torch.autograd.grad(p_b, inputs_b, grad_outputs=torch.ones_like(p_b), create_graph=True)[0]
                p_x_b, p_y_b, p_z_b = p_grad_b[:, 0], p_grad_b[:, 1], p_grad_b[:, 2]
                mask_x0 = (x_b == 0)
                mask_x1 = (x_b == 1)
                loss_bc_x = torch.mean(p_x_b[mask_x0]**2) + torch.mean(p_x_b[mask_x1]**2)
                mask_y0 = (y_b == 0)
                mask_y1 = (y_b == 1)
                loss_bc_y = torch.mean(p_y_b[mask_y0]**2) + torch.mean(p_y_b[mask_y1]**2)
                mask_z0 = (z_b == 0)
                mask_z1 = (z_b == 1)
                loss_bc_z = torch.mean(p_z_b[mask_z0]**2) + torch.mean(p_z_b[mask_z1]**2)
                loss_bc = loss_bc_x + loss_bc_y + loss_bc_z
                
                # Initial condition loss
                inputs_i = torch.stack([x_i, y_i, z_i, t_i], dim=1).to(x_r.device)
                p_i = model(inputs_i, p_scale, p_base)
                loss_ic = torch.mean(((p_i - p_base) / p_scale)**2)
                
                # Data loss
                inputs_d = torch.stack([x_d, y_d, z_d, t_d], dim=1).to(x_r.device)
                p_d_pred = model(inputs_d, p_scale, p_base)
                loss_data = torch.mean(((p_d_pred - p_d) / p_scale)**2)
                
                # Total loss with clamping to prevent NaN/inf
                loss = loss_pde + 100 * loss_bc + 10000 * loss_ic + 1000000 * loss_data
                loss = torch.clamp(loss, min=-1e10, max=1e10)  # Prevent numerical overflow

            # Check for NaN or inf in losses
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Epoch {epoch + 1}: Loss is NaN or inf, skipping...")
                continue

            # Backpropagation with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Store losses with NaN checking
            loss_pde_val = loss_pde.item() if torch.isfinite(loss_pde) else 0.0
            loss_bc_val = loss_bc.item() if torch.isfinite(loss_bc) else 0.0
            loss_ic_val = loss_ic.item() if torch.isfinite(loss_ic) else 0.0
            loss_data_val = loss_data.item() if torch.isfinite(loss_data) else 0.0
            loss_total_val = loss.item() if torch.isfinite(loss) else 0.0
            
            losses['pde'].append(loss_pde_val)
            losses['bc'].append(loss_bc_val)
            losses['ic'].append(loss_ic_val)
            losses['data'].append(loss_data_val)
            losses['total'].append(loss_total_val)
            
            # Update GUI and log to MLflow every 1000 epochs
            if (epoch + 1) % 1000 == 0 or epoch == 0:
                progress = (epoch + 1) / epochs
                progress_placeholder.progress(progress)
                epoch_placeholder.write(f"Epoch: {epoch + 1}/{epochs}")
                loss_placeholder.write(
                    f"**Losses**:\n"
                    f"- Total: {loss_total_val:.6f}\n"
                    f"- PDE: {loss_pde_val:.6f}\n"
                    f"- BC: {loss_bc_val:.6f}\n"
                    f"- IC: {loss_ic_val:.6f}\n"
                    f"- Data: {loss_data_val:.6f}"
                )
                mlflow.log_metric("total_loss", loss_total_val, step=epoch + 1)
                mlflow.log_metric("pde_loss", loss_pde_val, step=epoch + 1)
                mlflow.log_metric("bc_loss", loss_bc_val, step=epoch + 1)
                mlflow.log_metric("ic_loss", loss_ic_val, step=epoch + 1)
                mlflow.log_metric("data_loss", loss_data_val, step=epoch + 1)
                run_id_placeholder.write(f"MLflow Run ID: {run.info.run_id}")
            
            # Adaptive sampling
            if (epoch + 1) % 1000 == 0:
                residual_abs = torch.abs(residual)
                top_indices = torch.topk(residual_abs, k=1000).indices
                x_r = torch.cat([x_r, x_r[top_indices]])
                y_r = torch.cat([y_r, y_r[top_indices]])
                z_r = torch.cat([z_r, z_r[top_indices]])
                t_r = torch.cat([t_r, t_r[top_indices]])
        
        # Debug: Print losses before plotting
        print("Losses before plotting:", {key: len(losses[key]) for key in losses})
        for key in losses:
            print(f"{key} loss sample: {losses[key][:5]} ... {losses[key][-5:]}")
        
        # Save model and loss plot as artifacts
        model_path = "reservoir_pinn.pth"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
        mlflow.pytorch.log_model(model, "model")
        
        buf = plot_losses(losses)
        loss_plot_path = "loss_plot.png"
        with open(loss_plot_path, "wb") as f:
            f.write(buf.getvalue())
        mlflow.log_artifact(loss_plot_path)
    
    return losses

# Train model button
if st.button("Train Model"):
    if 'data_df' in st.session_state and st.session_state.data_df is not None:
        data = torch.tensor(st.session_state.data_df[['x', 'y', 'z', 't', 'p']].values, dtype=torch.float32).to(device)
        x_d, y_d, z_d, t_d, p_d = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
        N_d = len(data)
        N_r = 20000
        N_b = 2000
        N_i = 2000
        N_r_adaptive = 10000

        # Collocation points
        x_r = torch.rand(N_r).to(device)
        y_r = torch.rand(N_r).to(device)
        z_r = torch.rand(N_r).to(device)
        t_r = torch.rand(N_r).to(device)

        # Adaptive sampling near data points
        x_r_adaptive = torch.normal(mean=0.5, std=0.1, size=(N_r_adaptive,)).clamp(0, 1).to(device)
        y_r_adaptive = torch.normal(mean=0.5, std=0.1, size=(N_r_adaptive,)).clamp(0, 1).to(device)
        z_r_adaptive = torch.normal(mean=0.5, std=0.1, size=(N_r_adaptive,)).clamp(0, 1).to(device)
        t_r_adaptive = torch.rand(N_r_adaptive).to(device)

        x_r = torch.cat([x_r, x_r_adaptive])
        y_r = torch.cat([y_r, y_r_adaptive])
        z_r = torch.cat([z_r, z_r_adaptive])
        t_r = torch.cat([t_r, t_r_adaptive])

        # Boundary points (N_b = 2000, distributed across 6 faces)
        N_b_per_face = N_b // 6  # ~333 points per face
        remainder = N_b % 6      # Distribute remainder to ensure exact N_b
        face_counts = [N_b_per_face + 1 if i < remainder else N_b_per_face for i in range(6)]

        # x=0 and x=1 faces
        x_b = torch.cat([
            torch.zeros(face_counts[0]),  # x=0
            torch.ones(face_counts[1]),   # x=1
            torch.rand(face_counts[2]),   # y=0
            torch.rand(face_counts[3]),   # y=1
            torch.rand(face_counts[4]),   # z=0
            torch.rand(face_counts[5])    # z=1
        ]).to(device)

        # y=0 and y=1 faces
        y_b = torch.cat([
            torch.rand(face_counts[0]),   # x=0
            torch.rand(face_counts[1]),   # x=1
            torch.zeros(face_counts[2]),  # y=0
            torch.ones(face_counts[3]),   # y=1
            torch.rand(face_counts[4]),   # z=0
            torch.rand(face_counts[5])    # z=1
        ]).to(device)

        # z=0 and z=1 faces
        z_b = torch.cat([
            torch.rand(face_counts[0]),   # x=0
            torch.rand(face_counts[1]),   # x=1
            torch.rand(face_counts[2]),   # y=0
            torch.rand(face_counts[3]),   # y=1
            torch.zeros(face_counts[4]),  # z=0
            torch.ones(face_counts[5])    # z=1
        ]).to(device)

        # Random time for all boundary points
        t_b = torch.rand(N_b).to(device)

        # Initial condition points
        x_i = torch.rand(N_i).to(device)
        y_i = torch.rand(N_i).to(device)
        z_i = torch.rand(N_i).to(device)
        t_i = torch.zeros(N_i).to(device)

        with st.spinner("Training model..."):
            st.session_state.losses = train_model_with_updates(
                st.session_state.model, x_r, y_r, z_r, t_r, x_b, y_b, z_b, t_b, x_i, y_i, z_i, t_i, x_d, y_d, z_d, t_d, p_d, epochs,
                st.session_state.phi, st.session_state.k, st.session_state.mu, st.session_state.ct,
                st.session_state.q0, st.session_state.well_pos, st.session_state.p_scale, st.session_state.p_base
            )
            st.session_state.trained = True
            st.success("Model trained and saved to MLflow!")
            buf = plot_losses(st.session_state.losses)
            st.image(buf, caption="Training Losses")
    else:
        st.error("Please upload a valid CSV file with x, y, z, t, p columns before training.")

# Pressure Field Visualization
st.header("Visualize 3D Pressure Field Slice")
if st.session_state.trained and 'data_df' in st.session_state and st.session_state.data_df is not None:
    t_val = st.number_input("Select Time (t)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    z_slice = st.number_input("Select Z Slice", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    if st.button("Visualize Pressure Field"):
        with st.spinner("Generating pressure field slice..."):
            buf = plot_pressure_slice(
                st.session_state.model, t_val, st.session_state.data_df, device,
                z_slice, st.session_state.p_scale, st.session_state.p_base
            )
            st.image(buf, caption=f"Pressure Slice at z={z_slice}, t={t_val:.2f}")
else:
    st.info("Train or load a model and upload a valid CSV file to visualize the pressure field.")

# Pressure Field Animation
st.header("Generate 3D Pressure Field Animation")
if st.session_state.trained and 'data_df' in st.session_state and st.session_state.data_df is not None:
    z_slice_anim = st.number_input("Select Z Slice for Animation", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    if st.button("Generate Pressure Animation"):
        with st.spinner("Generating pressure field animation..."):
            animation_file = animate_pressure(
                st.session_state.model, st.session_state.data_df, device,
                z_slice_anim, st.session_state.p_scale, st.session_state.p_base, save_format='gif'
            )
            with open(animation_file, 'rb') as f:
                st.image(f.read(), caption="Pressure Animation (GIF)")
else:
    st.info("Train or load a model and upload a valid CSV file to generate the animation.")

# LLM Selection and Interaction
st.header("Query Pressure with LLM")
llm_options = ["DeepSeek", "OpenAI"] if openai_api_key else ["DeepSeek"]
st.session_state.llm_model = st.selectbox("Select LLM Model", llm_options, index=llm_options.index(st.session_state.llm_model))
if st.session_state.llm_model == "OpenAI" and not openai_api_key:
    st.warning("OpenAI API key not found in .env file. Using DeepSeek instead.")
    st.session_state.llm_model = "DeepSeek"

if st.session_state.trained:
    user_prompt = st.text_input("Ask about pressure (e.g., 'What is the pressure at x=0.5, y=0.5, z=0.5, t=0.5?')")
    if user_prompt:
        try:
            payload = {
                "prompt": user_prompt,
                "llm_model": st.session_state.llm_model
            }
            response = requests.post("http://localhost:8000/query", json=payload)
            if response.status_code == 200:
                result = response.json()
                if all(key in result for key in ['x', 'y', 'z', 't']):
                    x, y, z, t = float(result['x']), float(result['y']), float(result['z']), float(result['t'])
                    inputs = torch.tensor([[x, y, z, t]], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        pressure = st.session_state.model(inputs, st.session_state.p_scale, st.session_state.p_base).item()
                    st.write(f"Pressure at x={x}, y={y}, z={z}, t={t}: {pressure:.2f} Pa")
                else:
                    st.write(result.get('response', 'Could not extract coordinates.'))
            else:
                st.error("Error communicating with API server.")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Train or load a model first to query pressures.")


