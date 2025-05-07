import streamlit as st
import torch
import pandas as pd
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import os
import requests
from src.models.pinn import PINN
from src.utils.training import train_model, compute_pde_residual
from src.utils.plotting import plot_losses, plot_reservoir_data, plot_pressure_slice, animate_pressure

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY", "")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Streamlit app
st.title("3D Reservoir PINN Simulator")

# Sidebar for data visualization
with st.sidebar:
    st.header("3D Reservoir Data Visualization")
    if 'data_df' in st.session_state and st.session_state.data_df is not None:
        buf = plot_reservoir_data(st.session_state.data_df)
        st.image(buf, caption="3D Scatter Plot of Reservoir Data")
    else:
        st.info("Upload a CSV file with x, y, z, t, p columns to visualize data.")

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

# Input reservoir parameters
st.header("Reservoir Parameters")
col1, col2 = st.columns(2)
with col1:
    st.session_state.phi = st.number_input("Porosity (phi)", value=0.2, step=0.01)
    st.session_state.k = st.number_input("Permeability (k, Darcy)", value=0.1, step=0.01)
    st.session_state.mu = st.number_input("Viscosity (mu, cP)", value=1.0, step=0.1)
with col2:
    st.session_state.ct = st.number_input("Compressibility (ct, 1/Pa)", value=1e-6, step=1e-7, format="%.1e")
    st.session_state.q0 = st.number_input("Injection Rate (q0, m^3/s)", value=0.01, step=0.001)
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

# Input number of epochs
st.header("Training Parameters")
epochs = st.number_input("Number of Epochs", value=20000, step=1000, min_value=1000)

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
            losses = train_model(
                st.session_state.model, x_r, y_r, z_r, t_r, x_b, y_b, z_b, t_b, x_i, y_i, z_i, t_i, x_d, y_d, z_d, t_d, p_d, epochs,
                st.session_state.phi, st.session_state.k, st.session_state.mu, st.session_state.ct,
                st.session_state.q0, st.session_state.well_pos, st.session_state.p_scale, st.session_state.p_base
            )
            torch.save(st.session_state.model.state_dict(), 'reservoir_pinn.pth')
            st.session_state.trained = True
            st.success("Model trained and saved!")
            buf = plot_losses(losses)
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
    st.info("Train the model and upload a valid CSV file to visualize the pressure field.")

# Pressure Field Animation
st.header("Generate 3D Pressure Field Animation")
if st.session_state.trained and 'data_df' in st.session_state and st.session_state.data_df is not None:
    z_slice_anim = st.number_input("Select Z Slice for Animation", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    if st.button("Generate Pressure Animation"):
        with st.spinner("Generating pressure field animation..."):
            animation_file = animate_pressure(
                st.session_state.model, st.session_state.data_df, device,
                z_slice_anim, st.session_state.p_scale, st.session_state.p_base
            )
            with open(animation_file, 'rb') as f:
                st.video(f.read(), format="video/mp4")
else:
    st.info("Train the model and upload a valid CSV file to generate the animation.")

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
    st.info("Train the model first to query pressures.")

# Display PINN architecture
st.header("PINN Architecture")
try:
    img = Image.open('assets/pinn_architecture.png')
    st.image(img, caption="PINN Architecture")
except FileNotFoundError:
    st.warning("PINN architecture image not found. Run the notebook to generate it.")