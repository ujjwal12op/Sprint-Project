# # app.py
# import streamlit as st
# import subprocess
# import os
# import sys
# import pandas as pd
# import glob

# st.set_page_config(page_title="Crop Yield Prediction", layout="centered")

# st.title("🌾 Crop Yield Prediction UI")
# st.write("Run your ETL, train the model, and make predictions — all from here.")

# # Paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# SRC_DIR = os.path.join(BASE_DIR, "src")

# # Path to your venv's Python (ensures correct environment)
# VENV_PYTHON = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")

# # --- Step 1: Enrich Gold ---
# if st.button("1️⃣ Run Enrich Gold Pipeline"):
#     with st.spinner("Processing Silver → Gold..."):
#         subprocess.run(
#             [VENV_PYTHON, os.path.join(SRC_DIR, "enrich_gold.py")],
#             cwd=SRC_DIR
#         )
#     st.success("✅ Gold data created!")

# # --- Step 2: Train Model ---
# if st.button("2️⃣ Train Model"):
#     with st.spinner("Training Linear Regression Model..."):
#         result = subprocess.run(
#             [VENV_PYTHON, os.path.join(SRC_DIR, "train_model.py")],
#             cwd=SRC_DIR,
#             capture_output=True,
#             text=True
#         )
#     st.success("✅ Model trained and saved!")

#     # Try to extract RMSE and R² from script output
#     if result.stdout:
#         st.subheader("📈 Training Output")
#         st.code(result.stdout)
#         for line in result.stdout.splitlines():
#             if "RMSE" in line or "R²" in line:
#                 st.info(line)

# # --- Step 3: Predict ---
# if st.button("3️⃣ Predict"):
#     with st.spinner("Generating predictions..."):
#         subprocess.run(
#             [VENV_PYTHON, os.path.join(SRC_DIR, "predict.py")],
#             cwd=SRC_DIR
#         )
#     st.success("✅ Predictions saved!")

# # --- Optional: Show Predictions ---
# pred_dir = os.path.join(BASE_DIR, "data", "predictions", "crop_predictions")
# if os.path.exists(pred_dir):
#     st.subheader("📊 Latest Predictions")
#     st.write("Predictions are saved in parquet format.")
#     if st.button("View Predictions"):
#         files = glob.glob(os.path.join(pred_dir, "*.parquet"))
#         if files:
#             df = pd.read_parquet(files[0])
#             st.dataframe(df.head(20))

import streamlit as st
import subprocess
import os
import sys
import pandas as pd
import glob

st.set_page_config(page_title="Crop Yield Prediction", layout="centered")

st.title("🌾 Crop Yield Prediction UI")
st.write("Run your ETL, train the model, and make predictions — all from here.")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

# Always use the same Python that's running Streamlit
VENV_PYTHON = sys.executable

def run_script(script_name, capture=False):
    """Run a script from src/ with correct working directory."""
    return subprocess.run(
        [VENV_PYTHON, os.path.join(SRC_DIR, script_name)],
        cwd=SRC_DIR,
        capture_output=capture,
        text=True
    )

# --- Step 0: Extract Bronze ---
if st.button(" Extract Bronze Layer"):
    with st.spinner("Extracting Raw → Bronze..."):
        run_script("extract_bronze.py")
    st.success("✅ Bronze data created!")

if st.button(" Transform Silver Layer"):
    with st.spinner("Transforming silver layer..."):
        run_script("transform_silver.py")
    st.success("✅ data transformed!")

# --- Step 1: Enrich Gold ---
if st.button(" Run Enrich Gold Pipeline"):
    with st.spinner("Processing Silver → Gold..."):
        run_script("enrich_gold.py")
    st.success("✅ Gold data created!")

# --- Step 2: Train Model ---
if st.button(" Train Model"):
    with st.spinner("Training Linear Regression Model..."):
        result = run_script("train_model.py", capture=True)
    st.success("✅ Model trained and saved!")

    if result.stdout:
        st.subheader("📈 Training Output")
        st.code(result.stdout)
        for line in result.stdout.splitlines():
            if "RMSE" in line or "R²" in line:
                st.info(line)

# --- Step 3: Predict ---
if st.button("3️⃣ Predict"):
    with st.spinner("Generating predictions..."):
        run_script("predict.py")
    st.success("✅ Predictions saved!")

# --- Optional: Show Predictions ---
pred_dir = os.path.join(BASE_DIR, "data", "predictions", "crop_predictions")
if os.path.exists(pred_dir):
    st.subheader("📊 Latest Predictions")
    st.write("Predictions are saved in parquet format.")
    if st.button("View Predictions"):
        files = glob.glob(os.path.join(pred_dir, "*.parquet"))
        if files:
            df = pd.read_parquet(files[0])
            st.dataframe(df.head(20))