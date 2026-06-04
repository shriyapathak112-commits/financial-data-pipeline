import pennylane as qml
from pennylane import numpy as np
import scipy.optimize as opt

# ==========================================
# 1. Portfolio Dataset (10 Asset Universe)
# ==========================================
np.random.seed(42)
num_assets = 10

expected_returns = np.random.uniform(0.05, 0.20, num_assets)

# Symmetrical Covariance Matrix
raw_matrix = np.random.uniform(-0.02, 0.05, (num_assets, num_assets))
# Restructured scaling to ensure physical covariance balance
covariance_matrix = (raw_matrix + raw_matrix.T) / 2 + np.eye(num_assets) * 0.1
risk_aversion = 0.5 

# ==========================================
# 2. Hamiltonian Generator (FIXED LOGIC)
# ==========================================
def generate_portfolio_hamiltonian(returns, cov, q_lambda):
    """
    Converts Markowitz Portfolio Optimization to an Ising Hamiltonian.
    Correctly accumulates linear, quadratic, and constant shift terms.
    """
    obs = []
    coeffs = []
    
    # Initialize a constant scalar tracking identity energy offsets
    constant_shift = 0.0
    
    # 1. Process Diagonal Elements (Linear Returns & Self-Risk)
    for i in range(num_assets):
        # x_i = (1 - Z_i) / 2
        linear_weight = -returns[i] + q_lambda * cov[i, i]
        
        constant_shift += 0.5 * linear_weight
        
        coeffs.append(-0.5 * linear_weight)
        obs.append(qml.PauliZ(i))
        
    # 2. Process Off-Diagonal Elements (Cross-Asset Covariance)
    for i in range(num_assets):
        for j in range(i + 1, num_assets):
            # x_i * x_j = (1 - Z_i - Z_j + Z_i*Z_j) / 4
            # We multiply by 2 because cov[i,j] == cov[j,i] in symmetric matrices
            quad_weight = 2.0 * q_lambda * cov[i, j]
            
            constant_shift += 0.25 * quad_weight
            
            coeffs.append(-0.25 * quad_weight)
            obs.append(qml.PauliZ(i))
            
            coeffs.append(-0.25 * quad_weight)
            obs.append(qml.PauliZ(j))
            
            coeffs.append(0.25 * quad_weight)
            obs.append(qml.PauliZ(i) @ qml.PauliZ(j))
            
    # Add a single unified baseline Identity operator to the Hamiltonian
    if not np.isclose(constant_shift, 0.0):
        coeffs.append(constant_shift)
        obs.append(qml.Identity(0))
            
    return qml.Hamiltonian(coeffs, obs)

H = generate_portfolio_hamiltonian(expected_returns, covariance_matrix, risk_aversion)

# ==========================================
# 3. Quantum Device Configuration
# ==========================================
dev = qml.device("default.qubit", wires=num_assets)

# ==========================================
# 4. Ansatz Builder (Variational Circuit)
# ==========================================
def ansatz(params, wires):
    param_matrix = params.reshape(-1, len(wires))
    num_layers = param_matrix.shape[0]
    
    for layer in range(num_layers):
        for i, wire in enumerate(wires):
            qml.RY(param_matrix[layer, i], wires=wire)
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i+1]])

@qml.qnode(dev)
def cost_function(params):
    ansatz(params, wires=range(num_assets))
    return qml.expval(H)

# ==========================================
# 5. Optimization Loop & Result Extraction
# ==========================================
def run_vqe_optimization():
    num_layers = 3 # Increased layer depth slightly for better exploration space
    initial_params = np.random.uniform(0, 2 * np.pi, num_layers * num_assets)
    
    print("Launching Classical Optimizer VQE Loop...")
    # Switched to SLSQP for better constraint boundary behavior
    res = opt.minimize(cost_function, initial_params, method="SLSQP", options={"maxiter": 300})
    
    @qml.qnode(dev)
    def get_probs(params):
        ansatz(params, wires=range(num_assets))
        return qml.probs(wires=range(num_assets))
        
    probabilities = get_probs(res.x)
    best_bitstring_idx = np.argmax(probabilities)
    
    optimal_allocation = f"{best_bitstring_idx:0{num_assets}b}"
    
    print("\n==========================================")
    print("🎯 PORTFOLIO OPTIMIZATION RESULTS")
    print("==========================================")
    print(f"Optimal Asset Allocation Bitstring: {optimal_allocation}")
    for idx, selection in enumerate(optimal_allocation):
        status = "✅ INCLUDE" if selection == '1' else "❌ EXCLUDE"
        print(f"Asset {idx+1}: {status}")
    print("==========================================")

if __name__ == "__main__":
    run_vqe_optimization()
