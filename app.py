# Furniture arrangement in a small room (Input: room dimensions, furniture constraints → Output: optimized furniture placement).
import streamlit as st
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from deap import base, creator, tools, algorithms

# Generate synthetic dataset
def generate_synthetic_data():
    data = []
    for _ in range(50):
        room_size = (random.randint(5, 10), random.randint(5, 10))
        furniture_count = random.randint(3, 6)
        furniture = [
            (random.randint(1, 3), random.randint(1, 3)) for _ in range(furniture_count)
        ]
        target_layout = generate_target_layout(room_size, furniture)  # Optimize layout
        data.append((room_size, furniture, target_layout))
    return data

def generate_target_layout(room_size, furniture):
    # Simple heuristic: Place furniture in a row; just a placeholder for a more complex model
    layout = []
    x, y = 0, 0
    for w, h in furniture:
        layout.append((x, y, w, h))
        x += w
        if x >= room_size[0]:
            x = 0
            y += h
    return layout

data = generate_synthetic_data()

# Prepare data for training the machine learning model
def prepare_training_data(data):
    X = []
    y = []
    for room_size, furniture, target_layout in data:
        X.append(np.array(list(room_size) + [len(furniture)]))  # Convert room_size to list and append the number of furniture
        # Flatten target layout and ensure it has a consistent length
        flat_target = np.array([item for sublist in target_layout for item in sublist])
        y.append(flat_target)
    
    # Find the maximum length of the target layouts and pad them
    max_len = max(len(sample) for sample in y)
    y_padded = np.array([np.pad(sample, (0, max_len - len(sample))) for sample in y])
    
    return np.array(X), y_padded

X, y = prepare_training_data(data)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Streamlit UI for visualization
def visualize_layout(room_size, furniture, optimized_layout):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].set_title("Initial Layout")
    axes[1].set_title("Optimized Layout")
    
    # Visualization of the initial layout (furniture in the room)
    for ax, layout in zip(axes, [furniture, optimized_layout]):
        ax.set_xlim(0, room_size[0])
        ax.set_ylim(0, room_size[1])
        
        for idx, item in enumerate(layout):  # Ensure each item is unpacked correctly
            if len(item) == 2:  # Check if item has x, y only (for initial layout)
                x, y = item
                color = np.random.rand(3,)  # Random color for each furniture item
                ax.add_patch(plt.Rectangle((x, y), 1, 1, edgecolor='black', facecolor=color))  # Default width and height
            elif len(item) == 4:  # For optimized layout (x, y, w, h)
                x, y, w, h = item
                color = np.random.rand(3,)  # Random color for each optimized furniture item
                ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor='black', facecolor=color))
    st.pyplot(fig)

# Streamlit: Take user inputs
st.title("Room Layout Optimization")

# Set page background color to a blue-green mixture using inline CSS
st.markdown("""
    <style>
        body {
            background: linear-gradient(to top right, #00C9FF, #92FE9D); /* Gradient from blue to green */
            color: white;
            font-family: Arial, sans-serif;
        }
        .stButton>button {
            background-color: #4CAF50; /* Green */
            color: white;
            border-radius: 10px;
            padding: 10px;
            font-size: 18px;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #45a049; /* Slightly darker green */
        }
        .stMarkdown {
            color: white;
        }
        .stTitle {
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Input fields with validation to avoid zero values
room_width = st.number_input("Enter room width (1-50):", min_value=1, max_value=50, value=10)
room_height = st.number_input("Enter room height (1-50):", min_value=1, max_value=50, value=10)
furniture_count = st.number_input("Enter number of furniture items (1-20):", min_value=1, max_value=20, value=4)

# Validate room dimensions are greater than zero
if room_width <= 0 or room_height <= 0:
    st.error("Room dimensions must be greater than zero!")
else:
    # Color pickers for room and furniture colors
    room_color = st.color_picker("Pick a room color", "#ffffff")
    furniture_color = st.color_picker("Pick a furniture color", "#000000")

    # Generate random furniture sizes (within some range)
    furniture = [(random.randint(1, 3), random.randint(1, 3)) for _ in range(furniture_count)]

    # Button for layout generation with progress bar
    if st.button("Generate Layout"):
        progress = st.progress(0)
        for i in range(100):
            # Simulate layout generation process
            time.sleep(0.05)
            progress.progress(i + 1)
        
        features = np.array([room_width, room_height, furniture_count]).reshape(1, -1)  # Features based on user input
        optimized_layout = model.predict(features).reshape(-1, 4)  # Reshape to (x, y, w, h)

        # Visualize the room layout
        visualize_layout((room_width, room_height), furniture, optimized_layout)

        # Print layout description
        layout_description = "Initial Layout (Furniture Placement):\n"
        for i, (w, h) in enumerate(furniture):
            layout_description += f"Furniture {i+1}: Width = {w}, Height = {h}, Position = (x, y)\n"
        
        layout_description += "\nOptimized Layout (After AI Optimization):\n"
        for i, (x, y, w, h) in enumerate(optimized_layout):
            layout_description += f"Optimized Furniture {i+1}: Position = ({x:.2f}, {y:.2f}), Width = {w:.2f}, Height = {h:.2f}\n"

        st.text(layout_description)
