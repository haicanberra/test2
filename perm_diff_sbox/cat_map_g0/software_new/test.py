import numpy as np
from pylfsr import LFSR
import numpy as np
import dicom_global_params

'''

#for 5-bit LFSR with polynomial
seed = [1,1,0,1,0,0,1,1,1,0,0,0,1,1,1,1,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0]
fpoly = [1,2,3,31]
L = LFSR(fpoly=fpoly,initstate=seed, conf='galois',verbose=True)
seq = ""
for _ in range(10):
    L.next()
    seq = seq + str(L.outbit)
state = L.state
# L.next()
# seq = seq + str(L.outbit)

print(state)
# print(L.getSeq())
print(seq)
# L.seq = np.array([])
# print(L.getSeq())

# for _ in range(8):
#     L.next()
# state = L.state
# L.next()
# print(state)
# print(L.getSeq())

# new = 1 ^ 3 ^ 5
# new ^ 2 ^ 4

# P(X) =X^5+X^3+X^1+1
# L=5
# X^L.P(X^-1)  = X^5[X^-5+X^-3+X^-1+1] = X^5+X^4+X^2+1

#             1 2 3 4 5 
#            ┌─┬─┬─┬─┬─┐
#         ┌─→│0│1│1│0│1│─→
#         │  └┬┴─┴─┴─┴─┘
#         └──XOR──┘   │
#             └───XOR─┘ (taps == 1,3,5)

#for 5-bit LFSR with polynomial

seed = state
fpoly = [28,29,30,31]
L = LFSR(fpoly=fpoly,initstate=seed[::-1], conf='galois', verbose=True)
seq = ""
for _ in range(10):
    L.next()
    seq = seq + str(L.outbit)
state = L.state
# L.next()
# seq = seq + str(L.outbit)
print(state)
print(seq)

#             1 2 3 4 5 
#            ┌─┬─┬─┬─┬─┐
#         ┌─→│0│0│1│0│1│─→
#         │  └─┴┬┴─┴─┴─┘
#         └────XOR──┘ │
#               └─XOR─┘ (taps == 1,3,5)

'''

class LFSR:
    def __init__(self, seed, taps):
        self.state = seed
        self.taps = taps

    def next(self):
        feedback = 0
        for tap in self.taps:
            feedback ^= int(self.state[tap-1])
        self.state = str(feedback) + self.state[:-1]

        return feedback

    def output_bit(self):
        return self.state[-1]

    def get_state(self):
        return self.state
    
    def get_recovered_bit(self):
        return self.state[0]

# Example usage:
seed = "1101001110001111001101010100100"
taps = [1,2,3,31]

# Create a Galois-type LFSR
galois_lfsr = LFSR(seed, taps)

seq1 = ""
for _ in range(10):
    galois_lfsr.next()
    print("State:", galois_lfsr.get_state(), "Output Bit:", galois_lfsr.output_bit())
    seq1 = seq1 + galois_lfsr.output_bit()

print(seq1)

print("\n\n\n")

seed = galois_lfsr.get_state()[::-1]
print(seed)
taps = [28,29,30,31]

# Create a Galois-type LFSR
galois_lfsr = LFSR(seed, taps)

seq2 = ""
for _ in range(10):
    seq2 = seq2 + galois_lfsr.get_recovered_bit()
    galois_lfsr.next()
    print("State:", galois_lfsr.get_state()[::-1], "Recovered Bit:", galois_lfsr.get_recovered_bit())

print(seq2[::-1])