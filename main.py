import numpy as np

def discretize_spaces(x_step, u_step):
    # Bounds
    x_min, x_max = 0.0, 1.5
    u_min, u_max = -1.0, 1.0

    # Create discretized sets
    x_vals = np.arange(x_min, x_max + 1e-9, x_step)
    u_vals = np.arange(u_min, u_max + 1e-9, u_step)

    return x_vals, u_vals

# Example usage
if __name__ == "__main__":
    x_step = float(input("Enter step size for x: "))
    u_step = float(input("Enter step size for u: "))

    x_vals, u_vals = discretize_spaces(x_step, u_step)
    
    print("Discretized x:", x_vals)
    print("Discretized u:", u_vals)
