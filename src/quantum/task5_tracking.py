import os
import time
import mlflow
import matplotlib.pyplot as plt

# Connect to MLflow dashboard
mlflow.set_experiment("VQE_Portfolio_Optimization")

print("Starting Task 5 experiment tracking...")

with mlflow.start_run():
    # 1. Log tracking parameters
    mlflow.log_param("asset_count", 4)
    mlflow.log_param("ansatz_type", "TwoLocal")
    mlflow.log_param("optimizer", "SPSA")
    
    # --- Simulated VQE Process Block ---
    start_time = time.time()
    time.sleep(2) # Simulate execution time
    simulated_variance = 0.042
    simulated_iterations = 120
    end_time = time.time()
    # -----------------------------------
    
    # 2. Log final performance metrics
    mlflow.log_metric("portfolio_variance", simulated_variance)
    mlflow.log_metric("runtime", end_time - start_time)
    mlflow.log_metric("iterations", simulated_iterations)
    
    # 3. Generate visual loss plot
    plt.figure()
    plt.plot([0.5, 0.2, 0.1, 0.042], label="Loss convergence")
    plt.title("VQE Portfolio Optimization Loss")
    plt.legend()
    
    # Save chart locally and pass to MLflow artifact registry
    plot_name = "loss_plot.png"
    plt.savefig(plot_name)
    plt.close()
    
    mlflow.log_artifact(plot_name)
    
    # Cleanup local plot asset image
    if os.path.exists(plot_name):
        os.remove(plot_name)

print("Experiment logged successfully!")
