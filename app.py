import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

st.set_page_config(
    page_title="Multilayer Sphere Predictor",
    layout="wide"
)

st.title("Multilayer Sphere Neural-Network Predictor")

# ============================================================
# Load files
# ============================================================

@st.cache_resource
def load_model_and_scalers():
    model = tf.keras.models.load_model("final_multipole_model.keras")
    x_scaler = joblib.load("x_scaler.pkl")
    y_scaler = joblib.load("y_scaler.pkl")
    return model, x_scaler, y_scaler


@st.cache_data
def load_database():
    file_id = "1fWk9gZup7NfztnMRI9EarvA0Q_8Eyg_P"

    url = f"https://drive.google.com/uc?id={file_id}"

    df = pd.read_csv(url)

    return df

model, x_scaler, y_scaler = load_model_and_scalers()
df = load_database()

# ============================================================
# Columns
# ============================================================

target_cols = [
    "Cext_ED", "Cext_MD", "Cext_EQ", "Cext_MQ",
    "Cscat_ED", "Cscat_MD", "Cscat_EQ", "Cscat_MQ",
    "Cabs_ED", "Cabs_MD", "Cabs_EQ", "Cabs_MQ"
]

illumination_map = {
    0: "TEM",
    1: "TE",
    2: "TM"
}

# ============================================================
# Sidebar inputs
# ============================================================

st.sidebar.header("Input parameters")

AR = st.sidebar.number_input(
    "Aspect Ratio",
    value=float(df["AspectRatio"].min()),
    step=0.01
)

FF = st.sidebar.number_input(
    "Filling Factor",
    value=float(df["FillFactor"].min()),
    step=0.01
)

ILL = st.sidebar.selectbox(
    "Illumination",
    options=[0, 1, 2],
    format_func=lambda x: f"{x} ({illumination_map[x]})"
)

plot_button = st.sidebar.button("Plot Spectrum")

# ============================================================
# Functions
# ============================================================

def predict_spectrum(AR, FF, ILL, energy):

    X = np.column_stack([
        np.full(len(energy), AR),
        np.full(len(energy), FF),
        np.full(len(energy), ILL),
        energy
    ])

    X_scaled = x_scaler.transform(X)

    Y_scaled = model.predict(X_scaled, verbose=0)

    Y_pred = y_scaler.inverse_transform(Y_scaled)

    pred = pd.DataFrame(Y_pred, columns=target_cols)

    pred["Energy_eV"] = energy

    pred["Cext_total"] = pred[
        ["Cext_ED", "Cext_MD", "Cext_EQ", "Cext_MQ"]
    ].sum(axis=1)

    pred["Cscat_total"] = pred[
        ["Cscat_ED", "Cscat_MD", "Cscat_EQ", "Cscat_MQ"]
    ].sum(axis=1)

    pred["Cabs_total"] = pred[
        ["Cabs_ED", "Cabs_MD", "Cabs_EQ", "Cabs_MQ"]
    ].sum(axis=1)

    return pred


def get_real_data(AR, FF, ILL):

    real = df[
        np.isclose(df["AspectRatio"], AR) &
        np.isclose(df["FillFactor"], FF) &
        (df["Illumination"] == ILL)
    ].copy()

    if real.empty:
        return None

    real = real.sort_values("Energy_eV")

    real["Cext_total"] = real[
        ["Cext_ED", "Cext_MD", "Cext_EQ", "Cext_MQ"]
    ].sum(axis=1)

    real["Cscat_total"] = real[
        ["Cscat_ED", "Cscat_MD", "Cscat_EQ", "Cscat_MQ"]
    ].sum(axis=1)

    real["Cabs_total"] = real[
        ["Cabs_ED", "Cabs_MD", "Cabs_EQ", "Cabs_MQ"]
    ].sum(axis=1)

    return real


def make_plot(real, pred, quantity, title):

    fig, ax = plt.subplots(figsize=(6, 4))

    if real is not None:
        ax.plot(
            real["Energy_eV"],
            real[quantity],
            color="black",
            linewidth=2,
            label="Real"
        )

    ax.plot(
        pred["Energy_eV"],
        pred[quantity],
        color="red",
        linestyle="--",
        linewidth=2,
        label="Predicted"
    )

    ax.set_xlabel("Energy [eV]")
    ax.set_ylabel("Cross section")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig

# ============================================================
# Main app
# ============================================================

if plot_button:

    real = get_real_data(AR, FF, ILL)

    if real is not None:
        energy = real["Energy_eV"].values
        st.success("Real data found. Showing real vs predicted spectra.")
    else:
        energy = np.linspace(
            df["Energy_eV"].min(),
            df["Energy_eV"].max(),
            1000
        )
        st.warning("No exact real data found. Showing prediction only.")

    pred = predict_spectrum(AR, FF, ILL, energy)

    st.subheader(
        f"Aspect Ratio = {AR}, Filling Factor = {FF}, "
        f"Illumination = {ILL} ({illumination_map[ILL]})"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.pyplot(
            make_plot(real, pred, "Cext_total", "Total Extinction")
        )

    with col2:
        st.pyplot(
            make_plot(real, pred, "Cscat_total", "Total Scattering")
        )

    with col3:
        st.pyplot(
            make_plot(real, pred, "Cabs_total", "Total Absorption")
        )

else:
    st.info("Enter parameters in the sidebar and click Plot Spectrum.")