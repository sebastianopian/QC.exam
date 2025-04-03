from stabilizer import Pauli, CNOT, SWAP, H, S, X, Y, Z

class CliffordCircuit():

    def __init__(self, n:int):
        self.gates = []
        self.n = n

    def CNOT(self, control, target):
        self.gates.append(CNOT(control, target))
    
    def SWAP(self, control, target):
        self.gates.append(SWAP(control, target))
    
    def H(self, qubit):
        self.gates.append(H(qubit))

    def S(self, qubit):
        self.gates.append(S(qubit))

    def X(self, qubit):
        self.gates.append(X(qubit))

    def Y(self, qubit):
        self.gates.append(Y(qubit))

    def Z(self, qubit):
        self.gates.append(Z(qubit))
    
    def __call__(self, p:Pauli):
        for g in self.gates:
            p.apply(g)
        return p
    
    def __repr__(self):
        return repr([g.name for g in self.gates])