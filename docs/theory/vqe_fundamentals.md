# VQE Theory & Research Notes

## 1. VQE Architecture
The Variational Quantum Eigensolver (VQE) is a hybrid quantum-classical algorithm designed to find the ground state energy of a given Hamiltonian $H$. 

### Workflow Loop:
1. **Quantum State Preparation:** Prepare a parameterized quantum state $|\psi(\theta)\rangle$ using a specific ansatz.
2. **Measurement:** Measure the expectation value $\langle \psi(\theta) | H | \psi(\theta) \rangle$ on the quantum hardware.
3. **Classical Optimization:** Pass the energy value to a classical optimizer (e.g., COBYLA, SPSA) to compute updated parameters $\theta$.
4. **Iteration:** Repeat until convergence is achieved.

---

## 2. Financial Risk Minimization
In portfolio optimization, our goal is to map the classical portfolio variance (risk) equation to a problem Hamiltonian $H$. By minimizing the expectation value of this Hamiltonian, VQE finds the asset allocation strategy that yields the absolute lowest portfolio risk under given financial constraints.

---

## 3. Variational Circuits & Ansatz Design
An ansatz defines the structure of our parameterized quantum circuit. For this project, we analyze:
* **Hardware Efficient Ansatz (HEA):** Minimizes gate depth using native hardware gates but is susceptible to barren plateaus.
* **TwoLocal Ansatz:** Highly customizable modular circuit with alternating rotation and entanglement layers.
* **Custom Financial Ansatz:** Designed specifically to map the constraints of our 10-asset portfolio.

---

## 4. Quantum Gradients & The Parameter-Shift Rule
To execute gradient-based optimization on quantum hardware, we avoid classical finite differences and utilize the **Parameter-Shift Rule** for analytical gradients:

$$\frac{\partial \langle H \rangle}{\partial \theta} = \frac{\langle H \rangle_{\theta + \frac{\pi}{2}} - \langle H \rangle_{\theta - \frac{\pi}{2}}}{2}$$