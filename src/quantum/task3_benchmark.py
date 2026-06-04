import pennylane as qml
from pennylane import numpy as np
import scipy.optimize as opt
import time
import matplotlib.pyplot as plt

# ==========================================
# 1. Dataset Generation
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

# ==========================================
# 2. Ansatz Definitions
# ==========================================

def hardware_efficient_ansatz(params, wires):
    param_matrix = qml.math.reshape(params, (-1, len(wires)))
    for layer in range(param_matrix.shape[0]):
        for i, wire in enumerate(wires):
            qml.RY(param_matrix[layer, i], wires=wire)
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i+1]])

def twolocal_ansatz(params, wires):
    param_matrix = qml.math.reshape(params, (-1, 2, len(wires)))
    for layer in range(param_matrix.shape[0]):
        for i, wire in enumerate(wires):
            qml.RY(param_matrix[layer, 0, i], wires=wire)
            qml.RZ(param_matrix[layer, 1, i], wires=wire)
        for i in range(len(wires)):
            qml.CZ(wires=[wires[i], wires[(i + 1) % len(wires)]])

strong_risk_pairs = []
for i in range(num_assets):
    for j in range(i+1, num_assets):
        if abs(covariance_matrix[i, j]) > 0.01: 
            strong_risk_pairs.append((i, j))

def custom_financial_ansatz(params, wires):
    num_pairs = len(strong_risk_pairs)
    params_per_layer = len(wires) + num_pairs
    param_matrix = qml.math.reshape(params, (-1, params_per_layer))
    
    for layer in range(param_matrix.shape[0]):
        for i, wire in enumerate(wires):
            qml.RY(param_matrix[layer, i], wires=wire)
        for idx, (src, dst) in enumerate(strong_risk_pairs):
            p_idx = len(wires) + idx
            qml.IsingZZ(param_matrix[layer, p_idx], wires=[src, dst])

# ==========================================
# 3. Benchmark Execution Framework
# ==========================================
results_summary = {}

def run_benchmark(ansatz_name, ansatz_func, num_params_per_layer, num_layers=2):
    total_params = num_params_per_layer * num_layers
    initial_params = np.random.uniform(0, 2 * np.pi, total_params)
    
    cost_history = []
    
    @qml.qnode(dev)
    def cost_fn(params):
        ansatz_func(params, wires=range(num_assets))
        return qml.expval(H)
        
    # Fixed Tracking: capture values seamlessly inside objective pass
    def wrapped_objective(params):
        loss = cost_fn(params)
        cost_history.append(float(loss))
        return loss

    print(f"\n🚀 Running Benchmark for: {ansatz_name}...")
    start_time = time.time()
    
    res = opt.minimize(
        wrapped_objective, 
        initial_params, 
        method="Nelder-Mead", 
        options={"maxiter": 100}
    )
    
    runtime = time.time() - start_time
    
    @qml.qnode(dev)
    def get_probs(params):
        ansatz_func(params, wires=range(num_assets))
        return qml.probs(wires=range(num_assets))
        
    probs = get_probs(res.x)
    best_idx = np.argmax(probs)
    allocation_string = f"{best_idx:0{num_assets}b}"
    allocation_vector = np.array([int(b) for b in allocation_string])
    
    portfolio_risk = np.dot(allocation_vector.T, np.dot(covariance_matrix, allocation_vector))
    
    results_summary[ansatz_name] = {
        "history": cost_history,
        "runtime": runtime,
        "allocation": allocation_string,
        "risk": portfolio_risk,
        "final_cost": float(res.fun)
    }
    print(f"✅ Complete! Runtime: {runtime:.2f}s | Optimal Selection: {allocation_string}")

# Execute evaluations
run_benchmark("Hardware Efficient Ansatz", hardware_efficient_ansatz, num_params_per_layer=num_assets)
run_benchmark("TwoLocal Ansatz", twolocal_ansatz, num_params_per_layer=num_assets * 2)
run_benchmark("Custom Financial Ansatz", custom_financial_ansatz, num_params_per_layer=num_assets + len(strong_risk_pairs))

# ==========================================
# 4. Generate Performance Plots & Reports
# ==========================================
plt.figure(figsize=(10, 6))
for name, data in results_summary.items():
    plt.plot(data["history"], label=f"{name} (Final Cost: {data['final_cost']:.4f})", linewidth=2)

plt.title("Ansatz Architecture Convergence Comparison", fontsize=14, fontweight='bold')
plt.xlabel("Function Evaluations", fontsize=12)
plt.ylabel("Hamiltonian Expectation Energy (Cost)", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(fontsize=11)
plt.savefig("ansatz_convergence_comparison.png", dpi=300)
plt.show()

print("\n" + "="*50)
print(" FINAL COMPARISON REPORT DATA")
print("="*50)
print(f"{'Ansatz Structure':<30} | {'Runtime':<8} | {'Portfolio Risk':<15} | {'Allocation'}")
print("-"*75)
for name, data in results_summary.items():
    # Fixed string format presentation from .10 to standard string injection
    print(f"{name:<30} | {data['runtime']:.2f}s | {data['risk']:.5f} | {data['allocation']}")
print("="*50)