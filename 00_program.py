import qiskit as qc
from qiskit_aer import Aer
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Kraus, Operator, SparsePauliOp
from IPython.display import display, Latex

# Parametri
n = 4  # Numero di qubit
shots = 10000  # Numero di esecuzioni
L_reps = 2  # Numero di ripetizioni del circuito
p = 0.1  # Probabilità di applicare l'operatore di Kraus
h = 1  # Coefficiente nell'Hamiltoniano

# Funzione per generare parametri casuali
def generate_random_parameters(n):
    return 2 * np.pi * np.random.rand(2 * n)

# Funzione per creare la catena di CNOT come circuito
def create_cnot_chain_circuit(n):
    cnot_circuit = QuantumCircuit(n)
    for i in range(n - 1):
        cnot_circuit.cx(i, i + 1)
    return cnot_circuit

# Costruzione della matrice per la catena di CNOT
cnot_circuit = create_cnot_chain_circuit(n)
cnot_operator = Operator(cnot_circuit)

# Definizione degli operatori di Kraus
I_n = np.eye(2**n)  # Identità su n qubit
K0 = np.sqrt(1 - p) * I_n
K1 = np.sqrt(p) * cnot_operator.data  # Matrice della catena di CNOT
kraus_channel = Kraus([K0, K1])  # Creazione del canale di Kraus

# Creazione di W (Hamiltoniano)
pauli_strings = []
coefficients = []

for k in range(n):
    coeff = h * 2**(n / 2)  # Coefficiente del termine
    coefficients.append(coeff)

    # Stringa di Pauli per Z_k ⊗ Z_{k+1}
    pauli_string = ['I'] * n
    pauli_string[k] = 'Z'  # Z_k
    pauli_string[(k + 1) % n] = 'Z'  # Z_{k+1} (periodico)
    pauli_strings.append("".join(pauli_string))

# Creazione dell'Hamiltoniano come SparsePauliOp
W = SparsePauliOp(pauli_strings, coeffs=np.array(coefficients, dtype=complex))

# Mostra l'Hamiltoniano
print("Hamiltoniano W:")
print(W)

# Creazione del circuito quantistico
qc = QuantumCircuit(n, n)

# Ripeti il circuito L volte
for _ in range(L_reps):
    # Genera parametri casuali ad ogni iterazione
    parameters = generate_random_parameters(n)

    # Applica rotazioni
    for i in range(n):
        qc.rx(parameters[2 * i], i)
        qc.rz(parameters[2 * i + 1], i)

    # Applica il canale di Kraus
    qc.append(kraus_channel, list(range(n)))

    # Applica le operazioni CNOT personalizzate
    qc.cx(0, 1)
    qc.cx(3, 2)
    qc.cx(0, 2)
    qc.cx(2, 0)
    qc.cx(0, 2)
    qc.cx(1, 3)
    qc.cx(3, 1)
    qc.cx(1, 3)

# Creazione della matrice densità finale
rho = DensityMatrix.from_instruction(qc)

# Visualizzare la matrice densità finale
latex_rho = r"$\rho = \left(\begin{matrix}" + \
    f"{rho.data[0, 0]:.2f} & {rho.data[0, 1]:.2f} & \cdots & {rho.data[0, -1]:.2f} \\" + \
    f"{rho.data[1, 0]:.2f} & {rho.data[1, 1]:.2f} & \cdots & {rho.data[1, -1]:.2f} \\" + \
    r"\vdots & \vdots & \ddots & \vdots \\" + \
    f"{rho.data[-1, 0]:.2f} & {rho.data[-1, 1]:.2f} & \cdots & {rho.data[-1, -1]:.2f}" + \
    r"\end{matrix}\right)$"
display(Latex(latex_rho))

# Misura dei qubit
qc.measure(range(n), range(n))

# Simulazione
simulator = Aer.get_backend('aer_simulator')
qc = transpile(qc, simulator)
result = simulator.run(qc, shots=shots).result()

# Calcolo della traccia L = Tr(W * rho)
# Convertiamo W in una matrice densa e calcoliamo il prodotto con rho
W_matrix = W.to_matrix()
L = np.trace(np.dot(W_matrix, rho.data))

# Stampa del risultato
print(f"Il valore della traccia L è: {L:.5f}")

# Mostra il circuito
print("\nCircuito con la misura e il punto di calcolo di rho:")
print(qc.draw(output="text"))