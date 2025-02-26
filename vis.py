import numpy as np
import matplotlib.pyplot as plt

# For visualization, we create a simple 2D tensor (e.g., 32x32 matrix)
tensor_shape = (32, 32)

# Create a baseline tensor with all zeros
baseline_tensor = np.zeros(tensor_shape)

# -------------------------------
# INPUT16: Fault injection along the last dimension (columns)
# -------------------------------
# Make a copy for INPUT16 fault
input16_tensor = baseline_tensor.copy()
# Choose an arbitrary row to inject the fault
row_index = 15  
# Choose a starting column (ensure there's room for 16 consecutive elements)
start_col = 8  
# Inject fault: set 16 consecutive columns in the selected row to a nonzero value (e.g., 1)
input16_tensor[row_index, start_col:start_col+16] = 1

# -------------------------------
# WEIGHT16: Fault injection along the second-to-last dimension (rows)
# -------------------------------
# Make a copy for WEIGHT16 fault
weight16_tensor = baseline_tensor.copy()
# Choose an arbitrary column to inject the fault
col_index = 12  
# Choose a starting row (ensure there's room for 16 consecutive elements)
start_row = 10  
# Inject fault: set 16 consecutive rows in the selected column to a nonzero value (e.g., 1)
weight16_tensor[start_row:start_row+16, col_index] = 1

# -------------------------------
# Plotting the visualizations side-by-side
# -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot for INPUT16
axes[0].imshow(input16_tensor, cmap='hot', interpolation='nearest')
axes[0].set_title('INPUT16 Fault Injection\n(16 consecutive columns in one row)')
axes[0].set_xlabel('Columns')
axes[0].set_ylabel('Rows')

# Plot for WEIGHT16
axes[1].imshow(weight16_tensor, cmap='hot', interpolation='nearest')
axes[1].set_title('WEIGHT16 Fault Injection\n(16 consecutive rows in one column)')
axes[1].set_xlabel('Columns')
axes[1].set_ylabel('Rows')

plt.tight_layout()
plt.show()
