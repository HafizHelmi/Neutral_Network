import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Activation Functions Viewer", layout="centered")
st.title("Neural Network Activation Functions")
st.write("Explore how different activation functions introduce non-linearity into neural networks.")

app_mode = st.sidebar.radio(
    "Select Activation Function App:",
    ["1. ReLU (Rectified Linear Unit)", "2. Sigmoid", "3. Hyperbolic Tangent (Tanh)"]
)

x = np.linspace(-6, 6, 400)


if app_mode == "1. ReLU (Rectified Linear Unit)":
    st.subheader("1. Rectified Linear Unit (ReLU)")
    st.write("ReLU allows only positive values to pass through. Negative values are mapped to zero.")
    
  
    y_relu = np.maximum(0, x)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y_relu, label="ReLU", color="blue", linewidth=2)
    ax.set_title("ReLU Activation Function")
    ax.set_xlabel("Input (x)")
    ax.set_ylabel("Output (y)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


elif app_mode == "2. Sigmoid":
    st.subheader("2. Sigmoid Function")
    st.write("Sigmoid maps the input values to a probability range between 0 and 1.")
    
   
    y_sigmoid = 1 / (1 + np.exp(-x))
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y_sigmoid, label="Sigmoid", color="green", linewidth=2)
    ax.set_title("Sigmoid Activation Function")
    ax.set_xlabel("Input (x)")
    ax.set_ylabel("Output (y)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


elif app_mode == "3. Hyperbolic Tangent (Tanh)":
    st.subheader("3. Hyperbolic Tangent (Tanh)")
    st.write("Tanh is similar to the sigmoid function but maps the input to values between -1 and 1.")
    
    
    y_tanh = np.tanh(x)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y_tanh, label="Tanh", color="red", linewidth=2)
    ax.set_title("Tanh Activation Function")
    ax.set_xlabel("Input (x)")
    ax.set_ylabel("Output (y)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)