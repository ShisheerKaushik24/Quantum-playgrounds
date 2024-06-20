from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np
from qiskit.quantum_info import DensityMatrix, concurrence, state_fidelity
from qiskit import QuantumCircuit 

app = FastAPI()

@app.post("/calculate")
async def calculate_quantum_metrics(request: dict):
    temp_value = request['temp_value']
    gamma_0_values = request['gamma_0_values']
    time_values = request['time_values']

    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(0)
    qc.cx(0, 1)
    rho_0 = np.real(DensityMatrix(qc).data)

    b = 1

    def calculate_N(temp_value):
        return 1 / (np.exp(b / temp_value) - 1)

    def calculate_lambda(gamma_0, t):
        return 1 - np.exp(-gamma_0 * t)

    def calculate_gamma(gamma_0, N, t):
        return 1 - np.exp(-gamma_0 * (2 * N + 1) * t)

    N = calculate_N(temp_value)

    concurrence_results = {gamma_0: [] for gamma_0 in gamma_0_values}
    fidelity_results = {gamma_0: [] for gamma_0 in gamma_0_values}

    for gamma_0 in gamma_0_values:
        for t in time_values:
            lambda_val = calculate_lambda(gamma_0, t)
            K0 = np.array([[1, 0], [0, np.sqrt(1 - lambda_val)]])
            K1 = np.array([[0, np.sqrt(lambda_val)], [0, 0]])
            K = [K0, K1]

            E0 = np.array([[1, 0], [0, np.sqrt(1 - lambda_val)]])
            E1 = np.array([[0, np.sqrt(lambda_val)], [0, 0]])
            E = [E0, E1]

            rho_t = np.zeros_like(rho_0)

            for K_i in K:
                for E_j in E:
                    kron_product = np.kron(K_i, E_j)
                    kron_conj_transpose = np.conj(kron_product).T
                    rho_t += np.dot(kron_product, np.dot(rho_0, kron_conj_transpose))

            con_t = concurrence(rho_t)
            fidelity_t = state_fidelity(rho_0, rho_t) 
            
            concurrence_results[gamma_0].append(con_t)
            fidelity_results[gamma_0].append(fidelity_t)

    return JSONResponse(content={"concurrence": concurrence_results, "state_fidelity": fidelity_results})
