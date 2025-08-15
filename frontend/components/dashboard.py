import streamlit as st
import os

def show_dashboard():
    """Display model performance metrics and visualizations."""
    st.subheader("Model Performance Dashboard")
    metrics_image = "models/classification/training_metrics.png"
    
    if os.path.exists(metrics_image):
        st.image(metrics_image, caption="Training and Validation Loss/Accuracy", use_column_width=True)
        st.markdown("""
        **Metrics Description**:
        - **Train/Validation Loss**: Shows how the model's error decreases over epochs.
        - **Train/Validation Accuracy**: Shows classification accuracy on the training and validation sets (~85% validation accuracy).
        """)
    else:
        st.warning("Training metrics plot not found. Please run the training script to generate it.")
    
    # Placeholder for additional visualizations (e.g., class distribution)
    st.markdown("**More Visualizations Coming Soon**")
    st.write("Future additions: Class distribution, confusion matrix, etc.")