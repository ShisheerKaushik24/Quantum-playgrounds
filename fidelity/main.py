from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, state_fidelity
import numpy as np
import matplotlib.pyplot as plt

app = FastAPI()

class InputData(BaseModel):
    temp_value: float
    gamma_0_values: List[float]
    time_values: List[float]

@app.post("/calculate")

def calculate_state_fidelity(input_data: InputData):

    b = 1   
    temp_value = input_data.temp_value 
    gamma_0_values = input_data.gamma_0_values
    time_values = np.array(input_data.time_values)  

    def calculate_N(temp_value):
        return 1 / (np.exp((b) / (temp_value)) - 1)

    def calculate_lambda(gamma_0, t):
        result = 1 - np.exp(-gamma_0 * t)

    def calculate_gamma(gamma_0, N, t):
        return 1 - np.exp(-gamma_0 * (2 * N + 1) * t)

    def calculate_P(N):
        return (N + 1) / (2 * N + 1)

    N = calculate_N(temp_value)
    p = 1

    qc = QuantumCircuitBre(2)
    qc.x(0)
    qc.h(0)
    qc.cx(0, 1).cx(0, 1)

    rho_0 = np.real(DensityMatrix(qc).data)

    results = []

    for gamma_0 in gamma_0_values:
        lambda_values = calculate_lambda(gamma_0, time_values)
        gamma_values = calculate_gamma(gamma_0, N, time_values)
        concurrence_values = []

        for t, lambda_val, gamma_val in zip(time_values, lambda_values, gamma_values):

            # Define Kraus operators for the first type
            K0 = np.array([[1, 0], [0, np.sqrt(1 - lambda_val)]])
            K1 = np.array([[0, np.sqrt(lambda_val)], [0, 0]])
            K = [K0, K1]  # List of Kraus operators K_i

            # Define Kraus operators for the second type
            E0 = np.array([[1, 0], [0, np.sqrt(1 - lambda_val)]])
            E1 = np.array([[0, np.sqrt(lambda_val)], [0, 0]])
            E = [E0, E1] 

            # Initialize the result matrix rho(t) as a zero matrix of appropriate size
            rho_t = np.zeros_like(rho_0)

            # Iterate over all combinations of K and E
            for K_i in K:
                for E_j in E:
                    # Calculate the Kronecker product of K_i and E_j
                    kron_product = np.kron(K_i, E_j)

                    # Calculate the conjugate transpose (Hermitian) of K_i and E_j
                    kron_conj_transpose = np.conj(kron_product).T #np.kron(np.conj(K_i.T), np.conj(E_j.T))

                # Apply the formula: kron_product * rho_0 * kron_conj_transpose
                    rho_t += np.dot(kron_product, np.dot(rho_0, kron_conj_transpose))

            # Calculate concurrence for rho_t at current time step
            con_t = state_fidelity(rho_0, rho_t)
            concurrence_values.append(con_t)

        results.append({
            "gamma_0":gamma_0,
            "concurrence_values":concurrence_values
        })
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
