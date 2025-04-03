import numpy as np
from stabilizer import Pauli

class SingleQubitChannel():

    def __init__(self, t:dict, l:dict):
        self.t = t
        self.l = l
        self.index = {'X':0, 'Y':1, 'Z':2}
        self.heisenberg = True # True = Heisenberg, False = Schroedinger
    
    def _apply_schroedinger(self, p:Pauli, n:int):
        
        ps = p.to_string(ignore_phase=True)
        if n >= len(ps) or ps[n]=='I':

            plist = [(1.0, p)]
            for pi in 'XYZ':
                new_p = Pauli(ps[:n]+pi+ps[n+1:])
                new_i = self.index[pi]
                if self.t[new_i] > 1e-10:
                    plist.append((self.t[new_i], new_p))
            return plist
        
        i = self.index[ps[n]] 
        if self.l[i] > 1e-10: return [(self.l[i], p)]
        return []

    def _apply_heiseberg(self, p:Pauli, n:int):
        
        ps = p.to_string(ignore_phase=True)
        
        if n >= len(ps): return [(1.0, p)]
        if ps[n]=='I':  return [(1.0, p)]
        
        plist = []
        i = self.index[ps[n]]
        if self.l[i] > 1e-10: 
            plist.append((self.l[i], p))
        if self.t[i] > 1e-10:
            plist.append((self.t[i], Pauli(ps[:n]+'I'+ps[n+1:])))
        return plist

    def __call__(self, p, n):

        if self.heisenberg: return self._apply_heiseberg(p, n)
        return self._apply_schroedinger(p, n)

class AmplitudeDamping(SingleQubitChannel):

    def __init__(self, gamma:float):
        super().__init__(t=[0,0,gamma], l=[np.sqrt(1-gamma), np.sqrt(1-gamma), 1-gamma])

class Depolarizing(SingleQubitChannel):

    def __init__(self, p:float):
        super().__init__(t=[0,0,0], l=[1-p,1-p,1-p])


def _add_global_damping(ps:list[tuple], damping:float):
    
    damped_ps = []
    for d,p in ps:
        damped_ps.append((d*damping, p))
    return damped_ps

def single_qubit_noise_map(single_qubit_channels:list[tuple[int, SingleQubitChannel]]):

    def noise_map(p:Pauli):

        out = [(1.0, p)]
        for q, channel in single_qubit_channels:
            
            new_out = []
            for ci, pi in out:
                new_out += _add_global_damping(channel(pi, q) ,ci)
            out = new_out

        return out
    
    return noise_map
