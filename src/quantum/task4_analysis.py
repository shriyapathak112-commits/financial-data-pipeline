import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Dataset & Hamiltonian Setup
# ==========================================
np.random.seed(42)
num_assets = 10

expected_returns = np.random.uniform(0.05, 0.20, num_assets)
raw_matrix = np.random.uniform(-0.02, 0.05, (num_assets, num_assets))
covariance_matrix = (raw_matrix + raw_matrix.T) / 2 + np.eye(num_assets) * 0.1
risk_aversion = 0.5 

def generate_portfolio_hamiltonian(returns, cov, q_lambda):
    obs = []
    coeffs = []
    constant_shift = 0.0
    for i in range(num_assets):
        linear_weight = -returns[i] + q_lambda * cov[i, i]
        constant_shift += 0.5 * linear_weight
        coeffs.append(-0.5 * linear_weight)
        obs.append(qml.PauliZ(i))
    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            quad_weight = 2.0 * q_lambda * cov[i, j]
            constant_shift += 0.25 * quad_weight
            coeffs.append(-0.25 * quad_weight)
            obs.append(qml.PauliZ(i))
            coeffs.append(-0.25 * quad_weight)
            obs.append(qml.PauliZ(j))
            coeffs.append(0.25 * quad_weight)
            obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
    if not np.isclose(constant_shift, 0.0):
        coeffs.append(constant_shift)
        obs.append(qml.Identity(0))
    return qml.dot(coeffs, obs)

H = generate_portfolio_hamiltonian(expected_returns, covariance_matrix, risk_aversion)
dev = qml.device("default.qubit", wires=num_assets)

# Safe structural flattening configuration
num_layers = 2
def twolocal_ansatz(params, wires):
    # Fixed explicitly to handle dynamic tracking dimensions safely
    param_matrix = qml.math.reshape(params, (num_layers, 2, len(wires)))
    for layer in range(num_layers):
        for i, wire in enumerate(wires):
            qml.RY(param_matrix[layer, 0, i], wires=wire)
            qml.RZ(param_matrix[layer, 1, i], wires=wire)
        for i in range(len(wires)):
            qml.CZ(wires=[wires[i], wires[(i + 1) % len(wires)]])

@qml.qnode(dev)
def cost_function(params):
    twolocal_ansatz(params, wires=range(num_assets))
    return qml.expval(H)

# ==========================================
# 2. Optimization Loop with Metric Tracking
# ==========================================
num_params = num_layers * 2 * num_assets
params = np.random.uniform(0, np.pi, num_params, requires_grad=True)

loss_history = []
gradient_norm_history = []
parameter_evolution = []

opt = qml.AdamOptimizer(stepsize=0.05) # Lower stepsize for fine convergence trajectories
max_steps = 80

print("🚀 Starting learning behavior analysis...")

for step in range(max_steps):
    current_cost = cost_function(params)
    loss_history.append(float(current_cost))
    
    # Decouple data arrays safely
    parameter_evolution.append(np.array(params))
    
    # Capture analytical gradients
    grads = qml.grad(cost_function)(params)
    grad_norm = np.linalg.norm(grads)
    gradient_norm_history.append(float(grad_norm))
    
    # Update state
    params = opt.step(cost_function, params)
    
    if step % 10 == 0 or step == max_steps - 1:
        print(f" Step {step:02d} | Loss: {current_cost:.4f} | Gradient Norm: {grad_norm:.4f}")

parameter_evolution = np.array(parameter_evolution)
print("\n📊 Optimization finished. Generating visualizations...")

# ==========================================
# 3. Visualization Generator
# ==========================================
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Plot A: Loss Curves
axes[0].plot(loss_history, color='#1f77b4', linewidth=2.5, label='VQE Energy Loss')
axes[0].set_title('1. Loss Convergence Curve', fontsize=12, fontweight='bold')
axes[0].set_ylabel(r'Expectation Value $\langle H \rangle$', fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

# Plot B: Gradient Behavior (Norm)
axes[1].plot(gradient_norm_history, color='#d62728', linewidth=2, label='L2 Gradient Norm')
axes[1].set_title('2. Gradient Behavior over Time', fontsize=12, fontweight='bold')
axes[1].set_ylabel(r'$||\nabla \text{Cost}||$', fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()

# Plot C: Parameter Evolution
for i in range(5):
    axes[2].plot(parameter_evolution[:, i], label=r'Weight $\theta_{' + str(i) + r'}$')
axes[2].set_title('3. Parameter Evolution Trajectory (Subset)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Optimization Steps', fontsize=10)
axes[2].set_ylabel('Angle (Radians)', fontsize=10)
axes[2].grid(True, linestyle='--', alpha=0.5)
axes[2].legend(loc='upper right', ncol=5)

plt.tight_layout()
plt.savefig("vqe_convergence_analysis.png", dpi=300)
print("✅ Saved visualizations summary as: 'vqe_convergence_analysis.png'")
plt.show()