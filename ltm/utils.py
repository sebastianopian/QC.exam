import numpy as np

from stabilizer import Pauli
from ltm.clifford_circuit import CliffordCircuit


import itertools

def _add_identity(combination:str, bink:str):
    
    ps = ''
    for bit in bink:
        if bit == '1': ps += combination.pop(0)
        elif bit == '0': ps += 'I'
    return ps

def _block(p:Pauli):
    b = 0
    for i, pi in enumerate(reversed(p.to_string(ignore_phase=True))):
        if not (pi=='I'):
            b += 2**i
    return b

def _dimension_B(k:int):
    
    bink = bin(k)[2:]
    n = sum([int(bit) for bit in bink])
    return 3**n

def _B(k:int):
    
    bink = bin(k)[2:]
    n = sum([int(bit) for bit in bink])

    for combination in itertools.product('XYZ', repeat=n):
        pauli_description = _add_identity(list(combination), bink)
        yield Pauli(pauli_description)

def vardot(lv_1, lv_2):

    vd = 0
    for k,(l1, l2) in enumerate(zip(lv_1, lv_2)):
        vd += l1*l2/_dimension_B(k)
    return vd

def combination_of_pauli_lv(paulis:list[tuple[float, Pauli]], n:int):
    
    lv = np.zeros(2**n)
    for c,p in paulis:
        lv[_block(p)] += c**2
    return lv

def clifford_LTM(circ:CliffordCircuit, noise_map=None):

    if noise_map is None: noise_map = lambda x:[(1.0, x)]

    dim_LTM = 2**circ.n
    LTM = np.zeros(shape=(dim_LTM, dim_LTM))
    for k in range(dim_LTM):
        for p in _B(k):
            for damping, new_p in noise_map(circ(p)):
                LTM[_block(new_p),k] += damping**2/_dimension_B(k)
            
    return LTM