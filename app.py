import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_squared_error
import time

st.set_page_config(page_title="Data Processing Simulator", layout="wide", initial_sidebar_state="expanded")

st.title("Experimental Data Processing Simulator")
st.markdown("### Step-by-Step Data Cleaning: Generation, Averaging, and Interpolation")

st.sidebar.header("Data Source")
data_source = st.sidebar.radio("Select Data Source:", ["Simulate Data", "Upload CSV"])

st.sidebar.markdown("---")
st.sidebar.header("Processing Parameters")
savgol_window = st.sidebar.slider("Savitzky-Golay Window Length", 5, 51, 15, step=2)
savgol_poly = st.sidebar.slider("Savitzky-Golay Poly Order", 1, 5, 3)

if savgol_poly >= savgol_window:
    st.sidebar.warning("Poly order must be less than window length. Adjusting automatically.")
    savgol_poly = savgol_window - 1

has_ground_truth = False
mask = None

st.sidebar.markdown("---")
animate = st.sidebar.checkbox("Animate Graph Formation Live", value=True)
animation_speed = st.sidebar.slider("Animation Speed", 1, 5, 3) if animate else 0
anim_sleep = 0.5 / animation_speed if animate else 0.0

if data_source == "Simulate Data":
    st.sidebar.header("Simulation Parameters")
    signal_type = st.sidebar.selectbox("Base Signal Type", ["Sine Wave", "Decaying Sine", "Gaussian Pulse"])
    n_points = st.sidebar.slider("Number of Data Points", 50, 500, 200)
    noise_level = st.sidebar.slider("Noise Level (Std Dev)", 0.0, 1.0, 0.2)
    missing_pct = st.sidebar.slider("Percentage of Missing Data (%)", 0, 50, 20)

    def generate_base_signal(x, type_calc):
        if type_calc == "Sine Wave":
            return np.sin(2 * np.pi * x)
        elif type_calc == "Decaying Sine":
            return np.sin(4 * np.pi * x) * np.exp(-x * 2)
        elif type_calc == "Gaussian Pulse":
            return np.exp(-100 * (x - 0.5)**2)
        return np.zeros_like(x)

    x_true = np.linspace(0, 1, n_points)
    y_true = generate_base_signal(x_true, signal_type)
    
    np.random.seed(42)  # For reproducibility
    y_noisy_full = y_true + np.random.normal(0, noise_level, n_points)

    indices = np.arange(n_points)
    num_missing = int((missing_pct / 100.0) * n_points)
    
    if n_points > 20:
        missing_indices = np.random.choice(indices[10:-10], num_missing, replace=False)
    else:
        missing_indices = np.random.choice(indices[1:-1], num_missing, replace=False)
        
    missing_indices.sort()
    mask = np.ones(n_points, dtype=bool)
    mask[missing_indices] = False

    x_measured = x_true[mask]
    y_measured = y_noisy_full[mask]
    has_ground_truth = True
    
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV (First 2 columns: X, Y)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if len(df.columns) >= 2:
            x_measured = df.iloc[:, 0].values
            y_measured = df.iloc[:, 1].values
        else:
            x_measured = np.arange(len(df)).astype(float)
            y_measured = df.iloc[:, 0].values
            
        sort_idx = np.argsort(x_measured)
        x_measured = x_measured[sort_idx]
        y_measured = y_measured[sort_idx]
        
        x_true = np.linspace(x_measured.min(), x_measured.max(), len(x_measured)*3)
        y_true = None
        has_ground_truth = False
    else:
        st.info("Please upload a CSV file containing your noisy data to begin.")
        st.stop()

# Helper for fixed axes bounds
x_min, x_max = x_measured.min(), x_measured.max()
if has_ground_truth:
    x_min= min(x_min, x_true.min())
    x_max= max(x_max, x_true.max())

def get_y_range(*arrays):
    valid_arrays = [arr for arr in arrays if arr is not None and len(arr) > 0]
    min_y = min([np.min(arr) for arr in valid_arrays])
    max_y = max([np.max(arr) for arr in valid_arrays])
    padding = max(0.1 * abs(max_y - min_y), 0.1)
    return [min_y - padding, max_y + padding]

# Define how many chunks to break the animation into
chunks = 20 if animate else 1

# --- Visualization Step 1: Raw Noisy Data ---
with st.container():
    st.header("Step 1: Raw Noisy Experimental Data")
    if has_ground_truth:
        st.write("Generating a base signal, adding Gaussian random noise, and intentionally dropping points to simulate gaps.")
        yrange1 = get_y_range(y_true, y_measured)
    else:
        st.write("Visualizing your uploaded experimental data.")
        yrange1 = get_y_range(y_measured)

    chart1_placeholder = st.empty()
    
    for i in range(1, chunks + 1):
        idx_true = int((len(x_true) / chunks) * i)
        idx_meas = int((len(x_measured) / chunks) * i)

        fig1 = go.Figure()
        if has_ground_truth:
            fig1.add_trace(go.Scatter(x=x_true[:idx_true], y=y_true[:idx_true], mode='lines', name='True Signal', line=dict(color='rgba(0, 128, 0, 0.5)', dash='dash')))

        fig1.add_trace(go.Scatter(x=x_measured[:idx_meas], y=y_measured[:idx_meas], mode='markers', name='Noisy Measurements', marker=dict(color='red', size=5)))
        fig1.update_layout(title="Raw Data Overview", xaxis_title="Time / Distance", yaxis_title="Amplitude", template="plotly_white")
        fig1.update_xaxes(range=[x_min, x_max])
        fig1.update_yaxes(range=yrange1)
        
        chart1_placeholder.plotly_chart(fig1, use_container_width=True, key=f"step1_{i}")
        if animate and i < chunks:
            time.sleep(anim_sleep)


# --- Visualization Step 2: Averaging / Noise Reduction ---
with st.container():
    st.header("Step 2: Noise Reduction (Averaging)")
    st.write("Applying a **Savitzky-Golay filter** to mathematically smooth the noisy data while retaining the shape and height of waveform peaks.")
    st.write(f"Filter Parameters: Window Size = {savgol_window}, Polynomial Order = {savgol_poly}")

    actual_window = min(savgol_window, len(y_measured) - 1)
    if actual_window % 2 == 0:
        actual_window -= 1
    if actual_window <= savgol_poly:
        savgol_poly = actual_window - 1

    if actual_window > savgol_poly:
        y_smoothed = savgol_filter(y_measured, window_length=actual_window, polyorder=savgol_poly)
    else:
        st.error("Window length too small for filtering.")
        y_smoothed = y_measured

    yrange2 = get_y_range(y_measured, y_smoothed)
    chart2_placeholder = st.empty()
    
    for i in range(1, chunks + 1):
        idx_meas = int((len(x_measured) / chunks) * i)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=x_measured, y=y_measured, mode='markers', name='Noisy Backend Context', marker=dict(color='red', size=4), opacity=0.1)) # Show dim context instantly
        fig2.add_trace(go.Scatter(x=x_measured[:idx_meas], y=y_smoothed[:idx_meas], mode='lines+markers', name='Smoothed Data Forming', line=dict(color='blue', width=3), marker=dict(size=4)))
        fig2.update_layout(title="Noise Reduction via Savitzky-Golay Filter", xaxis_title="Time / Distance", yaxis_title="Amplitude", template="plotly_white")
        fig2.update_xaxes(range=[x_min, x_max])
        fig2.update_yaxes(range=yrange2)
        
        chart2_placeholder.plotly_chart(fig2, use_container_width=True, key=f"step2_{i}")
        if animate and i < chunks:
            time.sleep(anim_sleep)


# --- Visualization Step 3: Interpolation ---
with st.container():
    st.header("Step 3: Interpolation (Filling Gaps)")
    st.write("Using **Cubic Spline interpolation** to create a continuous curve predicting values across the entire domain.")

    try:
        cs = CubicSpline(x_measured, y_smoothed)
        y_interpolated_all = cs(x_true)
        
        yrange3 = get_y_range(y_smoothed, y_interpolated_all)
        chart3_placeholder = st.empty()

        for i in range(1, chunks + 1):
            idx_true = int((len(x_true) / chunks) * i)
            
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=x_measured, y=y_smoothed, mode='lines', name='Smoothed Context', line=dict(color='blue'), opacity=0.3))
            
            if has_ground_truth and mask is not None:
                # Need to find how many gap points are before current x_true
                current_x = x_true[idx_true-1] if idx_true > 0 else x_true[0]
                gap_x = x_true[~mask]
                valid_gaps = gap_x <= current_x
                y_interpolated_gaps = cs(gap_x[valid_gaps])
                
                fig3.add_trace(go.Scatter(x=gap_x[valid_gaps], y=y_interpolated_gaps, mode='markers', name='Filled Gaps Forming', marker=dict(color='orange', symbol='star', size=8)))
            else:
                fig3.add_trace(go.Scatter(x=x_true[:idx_true], y=y_interpolated_all[:idx_true], mode='lines', name='Cubic Spline Forming', line=dict(color='orange', dash='dash')))
                
            fig3.update_layout(title="Gap Filling via Cubic Spline Interpolation", xaxis_title="Time / Distance", yaxis_title="Amplitude", template="plotly_white")
            fig3.update_xaxes(range=[x_min, x_max])
            fig3.update_yaxes(range=yrange3)
            
            chart3_placeholder.plotly_chart(fig3, use_container_width=True, key=f"step3_{i}")
            if animate and i < chunks:
                time.sleep(anim_sleep)
                
    except Exception as e:
        st.error(f"Interpolation failed: {e}")
        y_interpolated_all = np.interp(x_true, x_measured, y_measured) # fallback to linear interpolation


# --- Visualization Step 4: Final Comparative Analysis ---
with st.container():
    st.header("Step 4: Final Comparative Analysis")
    fig4 = go.Figure()

    if has_ground_truth:
        fig4.add_trace(go.Scatter(x=x_true, y=y_true, mode='lines', name='Original True Signal', line=dict(color='green', width=2)))
        
    fig4.add_trace(go.Scatter(x=x_measured, y=y_measured, mode='markers', name='Raw Noisy Data', marker=dict(color='red', size=3), opacity=0.4))
    fig4.add_trace(go.Scatter(x=x_true, y=y_interpolated_all, mode='lines', name='Final Processed Signal', line=dict(color='purple', width=2)))
    fig4.update_layout(title="Final Processed Signal vs. Raw Data", xaxis_title="Time / Distance", yaxis_title="Amplitude", template="plotly_white")
    st.plotly_chart(fig4, use_container_width=True)

    # Metrics
    if has_ground_truth:
        st.subheader("Numerical Evaluation against Ground Truth")
        col1, col2 = st.columns(2)

        rmse_raw = np.sqrt(mean_squared_error(y_true[mask], y_measured))
        rmse_final = np.sqrt(mean_squared_error(y_true, y_interpolated_all))
        improvement = ((rmse_raw - rmse_final) / rmse_raw) * 100

        with col1:
            st.metric(label="RMSE (Raw Noisy vs True)", value=f"{rmse_raw:.4f}")

        with col2:
            st.metric(label="RMSE (Final Clean vs True)", value=f"{rmse_final:.4f}", delta=f"{improvement:.1f}% Improvement", delta_color="inverse")

        if improvement > 0:
            st.success("The pipeline successfully reduced the random noise and reconstructed missing sections accurately compared to the ground truth!")
        else:
            st.warning("The processed signal did not outperform the raw noisy data. Try adjusting the filter settings.")
    else:
        st.subheader("Processing Summary")
        st.write("Since uploaded data has no predefined 'Ground Truth', RMSE cannot be conclusively calculated. The final visualization demonstrates the effective separation of high-frequency noise and inference of intermediate data points.")

# --- Download Cleaned Data ---
st.markdown("---")
st.header("Export Results")
st.write("Download the final processed signal (after noise reduction and interpolation) as a CSV file.")

clean_df = pd.DataFrame({
    'Time / Distance (x)': x_true,
    'Cleaned Amplitude (y)': y_interpolated_all
})

csv_data = clean_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download Clean Data (CSV)",
    data=csv_data,
    file_name='cleaned_experimental_data.csv',
    mime='text/csv',
    type='primary'
)
