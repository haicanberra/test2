import math
import numpy as np
import os
import dicom_global_params
import dicom_cat_params
from dicom_image_functions import split_images, save_images
from fxpmath import Fxp

# Function: XOR two binary strings
# a, b: Operators (string)
# Return: a XOR b (string)
def xor(a, b):
    if (len(a) != len(b)):
        print("BIT XOR ERROR")
        return None
    
    ans = ""
     
    # Loop to iterate over the
    # Binary Strings
    for i in range(len(a)):
         
        # If the Character matches
        if (a[i] == b[i]):
            ans += "0"
        else:
            ans += "1"
    return ans


# Function: XOR list of binary strings
# mask: Mask, if mask[i] = 1, that means args[i] will be XORed, else if mask[i] = 0, args[i] will not be XORed (bit string)
# *args: List of input bit strings
# Return: mask[0] * args[0]  XOR  mask[1]  XOR  args[1] + ... + mask[K]  XOR  args[K]
def xor_list(mask, *args):
    result = "0" * dicom_global_params.PK_base_len
    for i in range(len(args)):
        if mask[i] == "1":
            result = xor(result, args[i])
    return result


# Function: DICOM MIE Bit Interleaving
# s1: Sequence 1
# s2: Sequence 2
# output_size: Size of output list
# Return: Bit string after Bit Manipulation
def bit_interleaving(s1, s2):
    # E1 = ''
    # E2 = ''
    
    # E1 = E1 + np.binary_repr(s1[0][0], width = math.ceil(math.log2(s1[0][0])))
    # E2 = E2 + np.binary_repr(s2[0][0], width = math.ceil(math.log2(s2[0][0])))

    E1 = str(s1)
    E2 = str(s2)
    
    if ((len(E1)) and (len(E2))):
        T = ''
        for i in range(len(E2)):
            T = T + E2[i] + E1[i]
    elif (len(E1)):
        T = str(E1)
    else:
        T = str(E2)
        
    return T


# Function: DICOM MIE Bit Manipulation with padded bit is '1'
# T: sequence
# output_size: Size of output list
# Return: Bit string after Bit Manipulation
def bit_manipulation_padd_one(T, output_size = 1):
    times = math.ceil(len(T) * 1.0 / output_size)
    E = ''
    if times <= 1.0: # |E1| + |E2| < |E|
        m = math.ceil(output_size * 1.0 / len(T)) # (m-1) * |T'| < |E| < m * |T'|
        T = T * int(m) # T = T' * m

        # Split E into n bit sequences
        n = math.ceil(len(T) * 1.0 / output_size) # (n-1) * |E| < |T| < n * |E|

        # Number of bit 0 needed to padd to Tn = |E| - |Tn|
        if ((len(T) % output_size) != 0):
            zero_length_pad = output_size * math.ceil(len(T) * 1.0 / output_size) - len(T) * 1.0
            str_zeros = '1' * int(zero_length_pad)
            T = T + str_zeros

        E = T[0:output_size]
        for i in range(1, n):
            E = xor(E, T[i*output_size:(i+1)*output_size])
        
    else: # |E1| + |E2| > |E|
        # Split E into n bit sequences
        n = math.ceil(len(T) * 1.0 / output_size) # (n-1) * |E| < |T| < n * |E|

        # Number of bit 0 needed to padd to Tn = |E| - |Tn|
        if ((len(T) % output_size) != 0):
            zero_length_pad = output_size * math.ceil(len(T) * 1.0 / output_size) - len(T) * 1.0
            str_zeros = '1' * int(zero_length_pad)
            T = T + str_zeros

        E = T[0:output_size]
        for i in range(1, n):
            E = xor(E, T[i*output_size:(i+1)*output_size])
        
    return E


# Function: Bit arrangement 1D to nD - from 1D matrix to nD matrix
# Y: Rule of bit arrangement (list of numpy arrays)
# B: Source bit-string array (string array-list of strings)
# Return: Destination bit-string array (string array or list of strings)
def bit_arrangement_1d_to_nd(Y: list, B) -> list:
    # Flip left/right verison of B
    B_fliplr = B.copy()
    
    matrix_B = []
    for i in range(len(B_fliplr)):
        # matrix_B[i] = [*B_fliplr[i]] # Unpack string to a list
        matrix_B.append([*B_fliplr[i]]) # Unpack string to a list

    # Y.shape
    width_Y = Y[0].shape[1]
    depth_Y = len(Y)

    A1 = []
    temp_A1 = []

    for k in range(depth_Y):
        for j in range(width_Y):
            if ((Y[k][0][j] == 0) & (Y[k][1][j] == 0)):
                temp_A1.append('0')
            elif ((Y[k][0][j] == 100) & (Y[k][1][j] == 100)):
                temp_A1.append('1')
            else:
                temp_A1.append(str(matrix_B[Y[k][0][j]-1][Y[k][1][j]-1]))
        A1.append(temp_A1)
        temp_A1 = []

    # Result: New string array (list of string)
    A = []
    for k in range(depth_Y):
        A.append(''.join(A1[k]))

    return A


# Function: Bit arrangement MIE nD
# Y: Rule of bit arrangement (list of numpy arrays)
# B: Source bit-string array (string array or list of strings)
# Return: Destination bit-string array (string array or list of strings)
def bit_arrangement_nd(Y: list, B) -> list:
    #Size of B
    height_B = len(B)
    # matrix_B = [] * height_B # [[''], [''],..., ['']]
    matrix_B = []
    for i in range(height_B):
        # matrix_B[i] = [*B[i]] # List of bits
        matrix_B.append([*B[i]]) # List of bits
    
    # Y.shape
    width_Y = Y[0].shape[1]
    depth_Y = len(Y)

    A1 = []
    temp_A1 = []

    for k in range(depth_Y):
        for j in range(width_Y):
            if ((Y[k][0][j] == 0) & (Y[k][1][j] == 0)):
                temp_A1.append('0')
            elif ((Y[k][0][j] == 100) & (Y[k][1][j] == 100)):
                temp_A1.append('1')
            else:
                temp_A1.append(str(matrix_B[Y[k][0][j]-1][Y[k][1][j]-1]))
        A1.append(temp_A1)
        temp_A1 = []

    # Result: New string array (list of string)
    A = []
    for k in range(depth_Y):
        A.append(''.join(A1[k]))

    return A


# Function: Session Key Scheduling
# S0: S0 (32-bit string)
# ID: ID (32-bit string)
# num_cycles: Number of cycles (integer)
# Return: 1. SK: List of secret keys (each key is it string with length = dicom_global_params.PK_base_len * 
#                                                                         dicom_global_params.K)
#         2. E: Input of PCM (string)
#         3. state: Current state (list)
def session_key_scheduling(lfsr, num_cycles):
    # Run num_cycle cycles
    seq = ""
    for _ in range(num_cycles):
        lfsr.next()
        seq = seq + str(lfsr.output_bit())
    state = lfsr.get_state()

    # E
    E = seq[:dicom_global_params.E_len]

    # SK
    SK_str = seq[dicom_global_params.E_len:]
    SK = [SK_str[i:i+dicom_global_params.pk_len] for i in range(0, len(SK_str), dicom_global_params.pk_len)]
    
    # print("\nseqqqqqqqqq: \n", seq, "\n")

    return E, SK, state


# Function: Session Key Scheduling for Decryption
# S0: S0 (32-bit string)
# ID: ID (32-bit string)
# num_cycles: Number of cycles (integer)
# Return: 1. SK: List of secret keys (each key is it string with length = dicom_global_params.PK_base_len * 
#                                                                         dicom_global_params.K)
#         2. E: Input of PCM (string)
#         3. state: Current state (list)
def session_key_scheduling_dec(lfsr, num_cycles):
    # Run num_cycle cycles
    seq = ""
    for _ in range(num_cycles):
        seq = seq + str(lfsr.get_recovered_bit())
        lfsr.next()
    state = lfsr.get_state()
    seq = seq[::-1]

    # E
    E = seq[:dicom_global_params.E_len]

    # SK
    SK_str = seq[dicom_global_params.E_len:]
    SK = [SK_str[i:i+dicom_global_params.pk_len] for i in range(0, len(SK_str), dicom_global_params.pk_len)]
    
    # print("\nseqqqqqqqqq: \n", seq, "\n")

    return E, SK, state


# Function: Session Key Scheduling for Decryption
# S0: S0 (32-bit string)
# ID: ID (32-bit string)
# num_cycles: Number of cycles (integer)
# Return: 1. SK: List of secret keys (each key is it string with length = dicom_global_params.PK_base_len * 
#                                                                         dicom_global_params.K)
#         2. E: Input of PCM (string)
#         3. state: Current state (list)
def session_key_scheduling_dec_k(lfsr, num_cycles, k):
    # Run num_cycle cycles
    seq = ""
    for _ in range(num_cycles):
        seq = seq + str(lfsr.get_recovered_bit())
        lfsr.next()
    state = lfsr.get_state()
    seq = seq[::-1]

    # E
    E = seq[:dicom_global_params.E_len]

    # SK
    SK_str = seq[dicom_global_params.E_len:]
    sk_k = [SK_str[k*dicom_global_params.pk_len:(k+1)*dicom_global_params.pk_len]]
    
    # print("\nseqqqqqqqqq: \n", seq, "\n")

    return E, sk_k, state


# Function: Save LFSR state
# n: Order of the current loop
# XY: Current [i, j]
# state: Current state of LFSR (list of lists)
# list_im_size: List of image sizes [(M_0, N_0), (M_1, N_1),..., (M_K, N_K)]
# Return: last_state (list of lists)
def save_lfsr_state(state, done, last_state):
    for k in range(dicom_global_params.K):
        if (done[k] != 1):
            last_state[k] = state[::-1]
    return last_state


# Function: Cat chaotic map
# gamma: gamma (fixed-point numbers array)
# xy: X_R (fixed-point numbers array - numpy array with dtype = Fxp)
# N: Number of bits to present xy (float)
# Return: Numpy array with dtype = Fxp
def cat_fi(gamma, xy: np.ndarray, N):
    xy_out = np.copy(xy)
    xy_out[0][0] = Fxp(xy[0][0] + gamma[0][0]*xy[1][0] - math.floor(xy[0][0] + gamma[0][0]*xy[1][0]), False, N, N-1)
    xy_out[1][0] = Fxp(gamma[1][0]*xy[0][0] + (gamma[0][0]*gamma[1][0] + 1)*xy[1][0] - math.floor(gamma[1][0]*xy[0][0] + (gamma[0][0]*gamma[1][0] + 1)*xy[1][0]), False, N, N-1)
    return xy_out


# Function: PCM Cat map
# E: Result after Bit Manipulation (string)
# Y_Gamma_0_FAST_Cat: Rule of bit arrangement (list of numpy arrays)
# Y_X_0_FAST_Cat: Rule of bit arrangement (list of numpy arrays)
# Y_FAST_Cat: Rule of bit arrangement (list of numpy arrays)
# R: Number of iterations (integer)
# Return: X_R (Numpy array with dtype = Fxp)
def pcm_cat(E: str, Y_Gamma_0_FAST_Cat, Y_X_0_FAST_Cat, Y_FAST_Cat, R):
    delta_gamma = bit_arrangement_nd(Y_Gamma_0_FAST_Cat, [E[0:16], E[16:32]])
    gamma_tmp = np.copy(dicom_cat_params.Gamma0_Cat)
    for i in range(len(gamma_tmp)):
        gamma_tmp_bin = xor(gamma_tmp[i][0].bin(), delta_gamma[i])
        gamma_tmp[i][0] = Fxp('0b' + gamma_tmp_bin, False, dicom_cat_params.m2_cat, dicom_cat_params.m2_cat - 6)

    delta_X = bit_arrangement_nd(Y_X_0_FAST_Cat, [E[32:48], E[48:64]])
    X_R_tmp = np.copy(dicom_cat_params.IV0_Cat)
    for i in range(len(X_R_tmp)):
        X_R_tmp_bin = xor(X_R_tmp[i][0].bin(), delta_X[i])
        X_R_tmp[i][0] = Fxp('0b' + X_R_tmp_bin, False, dicom_cat_params.m1_cat, dicom_cat_params.m1_cat - 1)

    for r in range(R):
        X_R = cat_fi(gamma_tmp, X_R_tmp, dicom_cat_params.m1_cat)

        delta_gamma = bit_arrangement_nd(Y_FAST_Cat, [X_R[0][0].bin(), X_R[1][0].bin()])
        for i in range(len(gamma_tmp)):
            gamma_tmp_bin = xor(gamma_tmp[i][0].bin(), delta_gamma[i])
            gamma_tmp[i][0] = Fxp('0b' + gamma_tmp_bin, False, dicom_cat_params.m2_cat, dicom_cat_params.m2_cat - 6)
    
    return X_R


# Function: Find new XY and phi_source, phi_dest to pass to Permutation and Diffusion (XYk for k = 1...K)
# X_R: X_R (numpy array with dtype = Fxp)
# SK: List of secret key
# list_im_size: List of image sizes [(M_0, N_0), (M_1, N_1),..., (M_K, N_K)]
# Return: 1. XY_new: New position [(i', j'), (i'', j''),...]
#         2. phi_source: After permutation, source plain pixel is XORed with phi_source
#         3. phi_dest: After permutation, destination plain pixel is XORed with phi_dest
def xy_phi_generation(X_R, SK, kP_plus, kC_minus, phi_source_base_position, phi_dest_base_position, i_base_position, j_base_position):
    # Generate new phi_source and phi_dest
    phi_source = []
    phi_dest   = []
    pc         = []
    for k in range(int(dicom_global_params.K/2)):
        bs_source_k = X_R[0][0].bin()[phi_source_base_position+2*k:phi_source_base_position+2*k+dicom_global_params.NB_max]
        bs_dest_k   = X_R[0][0].bin()[phi_dest_base_position+2*k:phi_dest_base_position+2*k+dicom_global_params.NB_max]

        phi_source_k = xor(bs_source_k, SK[k][-dicom_global_params.NB_max:])
        pc.append(bit_interleaving(s1 = np.binary_repr(kP_plus[k][0]).zfill(dicom_global_params.NB_max), s2 = np.binary_repr(kC_minus[k][0]).zfill(dicom_global_params.NB_max)))
        pc_k_manip   = bit_manipulation_padd_one(T = pc[k], output_size = dicom_global_params.NB_max)
        phi_source_k = xor(phi_source_k.zfill(dicom_global_params.NB_max), pc_k_manip)

        phi_dest_k = xor(bs_dest_k, SK[k][-2*dicom_global_params.NB_max:-dicom_global_params.NB_max])
        phi_dest_k = xor(phi_dest_k.zfill(dicom_global_params.NB_max), pc_k_manip)

        phi_source.append(phi_source_k)
        phi_dest.append(phi_dest_k)

    for k in range(int(dicom_global_params.K/2), dicom_global_params.K):
        k_ = k-int(dicom_global_params.K/2)
        
        bs_source_k = X_R[1][0].bin()[phi_source_base_position+1+2*k_:phi_source_base_position+1+2*k_+dicom_global_params.NB_max]
        bs_dest_k   = X_R[1][0].bin()[phi_dest_base_position+1+2*k_:phi_dest_base_position+1+2*k_+dicom_global_params.NB_max]

        phi_source_k = xor(bs_source_k, SK[k][-dicom_global_params.NB_max:])
        pc.append(bit_interleaving(s1 = np.binary_repr(kP_plus[k][0]).zfill(dicom_global_params.NB_max), s2 = np.binary_repr(kC_minus[k][0]).zfill(dicom_global_params.NB_max)))
        pc_k_manip   = bit_manipulation_padd_one(T = pc[k], output_size = dicom_global_params.NB_max)
        phi_source_k = xor(phi_source_k.zfill(dicom_global_params.NB_max), pc_k_manip)

        phi_dest_k = xor(bs_dest_k, SK[k][-2*dicom_global_params.NB_max:-dicom_global_params.NB_max])
        phi_dest_k = xor(phi_dest_k.zfill(dicom_global_params.NB_max), pc_k_manip)

        phi_source.append(phi_source_k)
        phi_dest.append(phi_dest_k)

    # Generate new i' and j'
    XY_new = []
    for k in range(int(dicom_global_params.K/2)):
        bs_X_k = X_R[0][0].bin()[i_base_position+2*k:i_base_position+2*k+int(math.ceil(math.log2(dicom_global_params.M_max)))]
        bs_Y_k = X_R[0][0].bin()[j_base_position+2*k:j_base_position+2*k+int(math.ceil(math.log2(dicom_global_params.N_max)))]

        X_new_k    = xor(bs_X_k, SK[k][0:int(math.ceil(math.log2(dicom_global_params.M_max)))])
        pc_k_manip = bit_manipulation_padd_one(T = pc[k], output_size = int(math.ceil(math.log2(dicom_global_params.M_max))))
        X_new_k    = xor(X_new_k, pc_k_manip)

        Y_new_k    = xor(bs_Y_k, SK[k][int(math.ceil(math.log2(dicom_global_params.M_max))):-2*dicom_global_params.NB_max])
        pc_k_manip = bit_manipulation_padd_one(T = pc[k], output_size = int(math.ceil(math.log2(dicom_global_params.N_max))))
        Y_new_k    = xor(Y_new_k, pc_k_manip)

        XY_new.append([X_new_k, Y_new_k])

    for k in range(int(dicom_global_params.K/2), dicom_global_params.K):
        k_ = k-int(dicom_global_params.K/2)

        bs_X_k = X_R[1][0].bin()[i_base_position+1+2*k_:i_base_position+1+2*k_+int(math.ceil(math.log2(dicom_global_params.M_max)))]
        bs_Y_k = X_R[1][0].bin()[j_base_position+1+2*k_:j_base_position+1+2*k_+int(math.ceil(math.log2(dicom_global_params.N_max)))]

        X_new_k    = xor(bs_X_k, SK[k][0:int(math.ceil(math.log2(dicom_global_params.M_max)))])
        pc_k_manip = bit_manipulation_padd_one(T = pc[k], output_size = int(math.ceil(math.log2(dicom_global_params.M_max))))
        X_new_k    = xor(X_new_k, pc_k_manip)

        Y_new_k    = xor(bs_Y_k, SK[k][int(math.ceil(math.log2(dicom_global_params.M_max))):-2*dicom_global_params.NB_max])
        pc_k_manip = bit_manipulation_padd_one(T = pc[k], output_size = int(math.ceil(math.log2(dicom_global_params.N_max))))
        Y_new_k    = xor(Y_new_k, pc_k_manip)

        XY_new.append([X_new_k, Y_new_k])

    return XY_new, phi_source, phi_dest


# Function: Find new XY and phi_source, phi_dest to pass to Permutation and Diffusion (XYk for k = 1...K)
# X_R: X_R (numpy array with dtype = Fxp)
# SK: List of secret key
# list_im_size: List of image sizes [(M_0, N_0), (M_1, N_1),..., (M_K, N_K)]
# list_num_bits_pre: List of number of bits to present a pixel of a channel
# Return: 1. XY_new: New position [(i', j'), (i'', j''),...]
#         2. phi_source: After permutation, source plain pixel is XORed with phi_source
#         3. phi_dest: After permutation, destination plain pixel is XORed with phi_dest
def xy_phi_generation_k(k, X_R, sk_k, kP_plus_k, kC_minus_k, phi_source_base_position_k, phi_dest_base_position_k, i_base_position_k, j_base_position_k):
    if (k <= dicom_global_params.K/2):
        bs_source_k = X_R[0][0].bin()[phi_source_base_position_k:phi_source_base_position_k+dicom_global_params.NB_max]
        bs_dest_k   = X_R[0][0].bin()[phi_dest_base_position_k:phi_dest_base_position_k+dicom_global_params.NB_max]

        bs_X_k      = X_R[0][0].bin()[i_base_position_k:i_base_position_k+int(math.ceil(math.log2(dicom_global_params.M_max)))]
        bs_Y_k      = X_R[0][0].bin()[j_base_position_k:j_base_position_k+int(math.ceil(math.log2(dicom_global_params.N_max)))]
    else:
        bs_source_k = X_R[1][0].bin()[phi_source_base_position_k:phi_source_base_position_k+dicom_global_params.NB_max]
        bs_dest_k   = X_R[1][0].bin()[phi_dest_base_position_k:phi_dest_base_position_k+dicom_global_params.NB_max]

        bs_X_k      = X_R[1][0].bin()[i_base_position_k:i_base_position_k+int(math.ceil(math.log2(dicom_global_params.M_max)))]
        bs_Y_k      = X_R[1][0].bin()[j_base_position_k:j_base_position_k+int(math.ceil(math.log2(dicom_global_params.N_max)))]

    phi_source_k = xor(bs_source_k, sk_k[-dicom_global_params.NB_max:])
    pc_k         = bit_interleaving(s1 = np.binary_repr(kP_plus_k[0]).zfill(dicom_global_params.NB_max), s2 = np.binary_repr(kC_minus_k[0]).zfill(dicom_global_params.NB_max))
    pc_k_manip   = bit_manipulation_padd_one(T = pc_k, output_size = dicom_global_params.NB_max)
    phi_source_k = xor(phi_source_k.zfill(dicom_global_params.NB_max), pc_k_manip)

    phi_dest_k = xor(bs_dest_k, sk_k[-2*dicom_global_params.NB_max:-dicom_global_params.NB_max])
    phi_dest_k = xor(phi_dest_k.zfill(dicom_global_params.NB_max), pc_k_manip)

    # XY_new_k
    X_new_k    = xor(bs_X_k, sk_k[0:int(math.ceil(math.log2(dicom_global_params.M_max)))])
    pc_k_manip = bit_manipulation_padd_one(T = pc_k, output_size = int(math.ceil(math.log2(dicom_global_params.M_max))))
    X_new_k    = xor(X_new_k, pc_k_manip)

    Y_new_k    = xor(bs_Y_k, sk_k[int(math.ceil(math.log2(dicom_global_params.M_max))):-2*dicom_global_params.NB_max])
    pc_k_manip = bit_manipulation_padd_one(T = pc_k, output_size = int(math.ceil(math.log2(dicom_global_params.N_max))))
    Y_new_k    = xor(Y_new_k, pc_k_manip)

    XY_new_k   = [X_new_k, Y_new_k]

    return XY_new_k, phi_source_k, phi_dest_k


# Function: Bit Pre-processing of the kth image
# XY_k: Current position i, j of kth image (type: [i, j])
# XY_new_k: New position (i, j) of kth image (type: [i, j])
# phi_source_k: After permutation, source plain pixel of kth image is XORed with phi_source_k
# phi_dest_k: After permutation, destination plain pixel of kth image is XORed with phi_dest_k
# Mk: Height of kth image
# Nk: Width of kth image
# num_bits_pre_k: Number of bits to present a pixel of a channel of the kth image
# Return: 1. XY_Pk: New [i, j] of the kth image
#         2. phi_source_Pk: phi_source_k after moding
#         3. phi_dest_Pk: phi_dest_k after moding
def bit_pre_processing_k(XY_k, XY_new_k, phi_source_k, phi_dest_k, M_k, N_k, num_bits_pre_k):
    # Split XY_new_k
    Y_new_k_bin = XY_new_k[1]
    # Y_new_k_bin = np.binary_repr(Y_new_k, width = math.ceil(math.log2(dicom_global_params.N_max)))

    X_new_k_bin = XY_new_k[0]
    # X_new_k_bin = np.binary_repr(X_new_k, width = math.ceil(math.log2(dicom_global_params.M_max)))

    # Find Y_Pk and X_Pk
    Y_Pk = Fxp(0, False, math.ceil(math.log2(dicom_global_params.N_max)), 0)
    # Y_Pk_bin = xor(Y_new_k_bin, ek[int(math.log2(dicom_global_params.M_max)):])
    Y_Pk.set_val('0b' + Y_new_k_bin)
    new_YPk = abs(Y_Pk.get_val() % 2**int(math.ceil(math.log2(N_k))) - N_k) if \
              Y_Pk.get_val() % 2**int(math.ceil(math.log2(N_k))) != 0 else N_k - 1
    Y_Pk.set_val(new_YPk)

    X_Pk = Fxp(0, False, math.ceil(math.log2(dicom_global_params.M_max)), 0)
    # X_Pk_bin = xor(X_new_k_bin, ek[0:int(math.log2(dicom_global_params.M_max))])
    X_Pk.set_val('0b' + X_new_k_bin)
    new_XPk = abs(X_Pk.get_val() % 2**int(math.ceil(math.log2(M_k))) - M_k) if \
              X_Pk.get_val() % 2**int(math.ceil(math.log2(M_k))) != 0 else M_k - 1
    X_Pk.set_val(new_XPk)

    # Check conditions
    XY_new_in_front_of_XY_1_pixel  = (((X_Pk == XY_k[0]) & (Y_Pk == XY_k[1] - 1))  | 
                                      ((X_Pk == XY_k[0] - 1) & (Y_Pk == N_k - 1) & (XY_k[1] == 0)) | 
                                      ((X_Pk == M_k - 1) & (Y_Pk == N_k - 1) & (XY_k[0] == 0) & (XY_k[1] == 0)))
    XY_new_after_XY_1_pixel        = (((X_Pk == XY_k[0]) & (Y_Pk == XY_k[1] + 1))  | 
                                      ((X_Pk == XY_k[0] + 1) & (Y_Pk == 0) & (XY_k[1] == N_k - 1)) | 
                                      ((X_Pk == 0) & (Y_Pk == 0) & (XY_k[0] == M_k - 1) & (XY_k[1] == N_k - 1)))
    XY_new_is_XY                   = ((X_Pk  == XY_k[0]) & (Y_Pk == XY_k[1]     ))
    XY_new_in_front_of_XY_2_pixels = (((X_Pk == XY_k[0]) & (Y_Pk == XY_k[1] - 2 )) | 
                                      ((X_Pk == XY_k[0] - 1) & (Y_Pk == N_k - 1) & (XY_k[1] == 1)) | 
                                      ((X_Pk == XY_k[0] - 1) & (Y_Pk == N_k - 2) & (XY_k[1] == 0)) | 
                                      ((X_Pk == M_k - 1) & (Y_Pk == N_k - 2) & (XY_k[0] == 0) & (XY_k[1] == 0)) | 
                                      ((X_Pk == M_k - 1) & (Y_Pk == N_k - 1) & (XY_k[0] == 0) & (XY_k[1] == 1)))
    XY_new_after_XY_2_pixels       = (((X_Pk == XY_k[0]) & (Y_Pk == XY_k[1] + 2 )) | 
                                      ((X_Pk == XY_k[0] + 1) & (Y_Pk == 0) & (XY_k[1] == N_k - 2)) | 
                                      ((X_Pk == XY_k[0] + 1) & (Y_Pk == 1) & (XY_k[1] == N_k - 1)) | 
                                      ((X_Pk == 0) & (Y_Pk == 0) & (XY_k[0] == M_k - 1) & (XY_k[1] == N_k - 2)) | 
                                      ((X_Pk == 0) & (Y_Pk == 1) & (XY_k[0] == M_k - 1) & (XY_k[1] == N_k - 1)))
    
    if ((XY_new_in_front_of_XY_1_pixel | XY_new_after_XY_1_pixel | XY_new_is_XY | XY_new_in_front_of_XY_2_pixels | XY_new_after_XY_2_pixels)):
        if (X_Pk < M_k - 1):
            X_Pk = X_Pk + 1
        else:
            X_Pk = X_Pk - 1
    
    # XY_Pk
    XY_Pk = [np.uint16(X_Pk), np.uint16(Y_Pk)]

    # phi_source_Pk
    phi_source_Pk = phi_source_k[-num_bits_pre_k:].zfill(dicom_global_params.NB_max)

    # phi_dest_Pk
    phi_dest_Pk = phi_dest_k[-num_bits_pre_k:].zfill(dicom_global_params.NB_max)

    return XY_Pk, phi_source_Pk, phi_dest_Pk


# Function: Bit Pre-processing of all image
# ID: List of User IDs and Image IDs ([[user_id_1, image_id_0], [user_id_1, image_id_1], [user_id_1, image_id_2], ...])
# XY: Current i and j ([(i', j'), (i'', j''),...])
# XY_new: New position [(i_new', j_new'), (i_new'', j_new''),...]
# phi_source: After permutation, source plain pixels are XORed with phi_source
# phi_dest: After permutation, destination plain pixels are XORed with phi_dest
# PCF: PCF
# list_im_size: List of image sizes [(M_0, N_0), (M_1, N_1),..., (M_K, N_K)]
# list_num_bits_pre: List of number of bits to present a pixel of a channel
# Return: 1. XY_P: New (i, j) after moding ([[i_new_0, j_new_0], [i_new_1, j_new_1], ...])
#         2. phi_source_P: phi_source after moding
#         3. phi_dest_P: phi_dest after moding
def bit_pre_processing(XY, XY_new, phi_source, phi_dest, list_im_size, list_num_bits_pre):
    XY_P = []
    phi_source_P = []
    phi_dest_P = []
    for k in range(dicom_global_params.K):
        XY_Pk, phi_source_Pk, phi_dest_Pk = bit_pre_processing_k(XY_k           = [XY[k][0]    , XY[k][1]    ], 
                                                                 XY_new_k       = [XY_new[k][0], XY_new[k][1]], 
                                                                 phi_source_k   = phi_source[k]               , 
                                                                 phi_dest_k     = phi_dest[k]                 , 
                                                                 M_k            = list_im_size[k][0]          , 
                                                                 N_k            = list_im_size[k][1]          , 
                                                                 num_bits_pre_k = list_num_bits_pre[k]         )
        XY_P.append(XY_Pk)
        phi_source_P.append(phi_source_Pk)
        phi_dest_P.append(phi_dest_Pk)

    return XY_P, phi_source_P, phi_dest_P


# Function: S-box sub byte
# Sbox: S-box (tuple)
# bit_sequence: Bit sequence (string)
# Return: Bit sequence after processing with S-box
def sbox_subbyte(Sbox, bit_sequence):
    if dicom_global_params.NB_max == 24:
        result_highest = Sbox[int(bit_sequence[0:8], 2)]
        result_high    = Sbox[int(bit_sequence[8:16], 2)]
        result_low     = Sbox[int(bit_sequence[16:24], 2)]
        return (np.binary_repr(result_highest, width = 8) + np.binary_repr(result_high, width = 8) + np.binary_repr(result_low, width = 8))
    else:
        return np.binary_repr(Sbox[int(bit_sequence, 2)], width = 8)


# Function: DICOM MIE Permutation and Diffusion of Encryption processing
# kI: Images (list of image matrices)
# XY: Current i and j ([(i', j'), (i'', j''),...])
# XY_P: New position [(i_new', j_new'), (i_new'', j_new''),...]
# phi_source_P: After permutation, source plain pixel is XORed with phi_source_P
# phi_dest_P: After permutation, destination plain pixel is XORed with phi_dest_P
# prev_kP_plus: Previous kP+
# kC_minus: kC- (the same type as kC0 = np.asarray([[123], [11 ], [27], [88], [33], [211], [97], [63]]))
# list_im_size: List of image sizes [(M_0, N_0), (M_1, N_1),..., (M_K, N_K)]
# n: Current iteration's order
# Return: 1. kC_to_pcm: Pixel's value used to impact chaotic map (the same type as kC_minus)
#         2. kP_plus: kP+
#         3. kI: Images after Permutation and Diffusion (list of image matrices)
def dicom_mie_perm_diff_enc(kI, XY, XY_P, phi_source_P, phi_dest_P, prev_kP_plus, kC_minus, Sbox, list_im_size, list_num_bits_pre, list_n, done, dir_cipher, str_Fnames, grey_image):
    # For the next pixel: Pass the diffused pixel's value to chaotic map
    kC_to_pcm = np.zeros_like(kC_minus, dtype=np.uint32)

    # Find kP_plus for the next pixel
    kP_plus = np.zeros((dicom_global_params.K, 1), dtype=np.uint32)

    for idx in range(dicom_global_params.K):
        i = XY[idx][0]
        j = XY[idx][1]

        if (done[idx] != 1):
            # Permutation
            temp = kI[idx][i][j]
            kI[idx][i][j] = kI[idx][XY_P[idx][0]][XY_P[idx][1]]
            kI[idx][XY_P[idx][0]][XY_P[idx][1]] = temp
            
            # Diffusion and S-box sub byte
            temp = kI[idx][i][j] # Current pixel after Permutation
            temp_str = xor(np.binary_repr(temp, width = dicom_global_params.NB_max), np.binary_repr(kC_minus[idx][0], width = dicom_global_params.NB_max)) # temp_value = I[i][j] XOR C[i-1][j]
            temp_str = xor(temp_str, phi_source_P[idx]) # temp_value XOR phi_source_P (result of chaotic map)
            kI[idx][i][j] = np.uint32(int(temp_str, 2))
            
            # The pixel permuted with current pixel is also diffused
            temp = kI[idx][XY_P[idx][0]][XY_P[idx][1]]
            temp = xor(np.binary_repr(temp, width = dicom_global_params.NB_max), phi_dest_P[idx])
            kI[idx][XY_P[idx][0]][XY_P[idx][1]] = np.uint32(int(temp, 2))


            if ((i == list_im_size[idx][0] - 1) & (j == list_im_size[idx][1] - 1)):
                # S-box for all pixels
                for height in range(list_im_size[idx][0]):
                    for width in range(list_im_size[idx][1]):
                        temp = kI[idx][height][width]
                        temp_str = sbox_subbyte(Sbox = Sbox, bit_sequence = np.binary_repr(temp, width = dicom_global_params.NB_max))
                        kI[idx][height][width] = np.uint32(int(temp_str, 2))


            # For the next pixel: Pass the diffused pixel's value (kC_minus) to chaotic map
            kC_to_pcm[idx][0] = kI[idx][i][j]


            # Find kP_plus for the next pixel
            if (i < list_im_size[idx][0] - 1): # This means i + 1 is not out of range (i + 1 <= M_k - 1)
                if (j < list_im_size[idx][1] - 2): # This means k[i][j+2] is inside image because j + 2 is not out of range (j + 2 <= N_k - 1)
                    kP_plus[idx][0] = np.uint32(kI[idx][i][j+2])
                else: # This means j + 2 is out of range (j + 2 > N_k - 1) - gradually to the end of the line
                    kP_plus[idx][0] = np.uint32(kI[idx][i+1][j-(list_im_size[idx][1]-1)+1]) # Go to the (j-(N_k-1)+1)th pixel of the next rowsd
            else: # This means i + 1 is out of range (i + 1 > M_k - 1)
                if (j < list_im_size[idx][1] - 2): # This means j + 2 is not out of range (j + 2 <= N_k - 1)
                    kP_plus[idx][0] = np.uint32(kI[idx][i][j+2])
                elif(j == list_im_size[idx][1] - 2): # This mean j + 2 == N_k
                    if (list_n[idx] == dicom_global_params.Ne - 1): # The last iteration
                        kP_plus[idx][0] = np.uint32(dicom_global_params.kP0[idx][0] & (2**list_num_bits_pre[idx] - 1))
                    else: # Not the last iteration
                        kP_plus[idx][0] = np.uint32(kI[idx][0][0])
                else: # This means j == N_k - 1
                    if (list_n[idx] < dicom_global_params.Ne - 1): # Not the last iteration
                        kP_plus[idx][0] = np.uint32(kI[idx][0][1])

            if (i < list_im_size[idx][0] - 1):
                if (j < list_im_size[idx][1] - 1):
                    XY[idx] = [i, j + 1]
                else:
                    XY[idx] = [i + 1, 0]
            else:
                if (j < list_im_size[idx][1] - 1):
                    XY[idx] = [i, j + 1]
                else:
                    kC_merge = split_images(kI                = [kI[idx]]               , 
                                            list_num_bits_pre = [list_num_bits_pre[idx]] )

                    save_images(kC          = kC_merge                           , 
                                folder_path = dir_cipher + "_" + str(list_n[idx]), 
                                str_Fnames  = [str_Fnames[idx]]                  , 
                                grey_image  = grey_image                          )
                    
                    if (list_n[idx] == dicom_global_params.Ne - 1):
                        XY[idx]       = [0, 0]
                        done[idx]     = 1
                    else:
                        XY[idx]       = [0, 0]
                        list_n[idx]   = list_n[idx] + 1
        else:
            kP_plus[idx][0] = prev_kP_plus[idx][0]
            kC_to_pcm[idx][0] = kC_minus[idx][0]

    return XY, kC_to_pcm, kP_plus, kI


# Function: MIE Permutation and Diffusion of Decryption processing
# ID: List of User IDs and Image IDs ([[user_id_1, image_id_0], [user_id_1, image_id_1], [user_id_1, image_id_2], ...])
# kC: Cipher images (list of image matrices)
# XY: Current i and j (list [i, j])
# XY_P: New position [(i', j'), (i'', j''),...]
# phi_source_P: After permutation, source plain pixel is XORed with phi_source_P
# phi_dest_P: After permutation, destination plain pixel is XORed with phi_dest_P
# prev_kP_plus: Previous kP+
# kC_minus: kC- (the same type as kC0 = np.asarray([[123], [11 ], [27], [88], [33], [211], [97], [63]]))
# list_im_size: List of image sizes [(M_0, N_0), (M_1, N_1),..., (M_K, N_K)]
# n: Current iteration's order
# Return: 1. kC_minus_next: kC- for the next pixel
#         2. kP_to_pcm: Pixel's value used to impact chaotic map (the same type as kC_minus)
#         3. kC: kC
def dicom_mie_perm_diff_dec(kC, XY, XY_P, phi_source_P, phi_dest_P, prev_kP_plus, kC_minus, Sbox, list_im_size, list_num_bits_pre, list_n, iter, done):
    # For the next pixel: Pass the diffused pixel's value to chaotic map
    kP_to_pcm = np.zeros_like(kC_minus, dtype=np.uint32)

    # Find kC_minus for the next pixel
    kC_minus_next = np.zeros((dicom_global_params.K, 1), dtype=np.uint32)

    for idx in range(dicom_global_params.K - 1, -1, -1):
        i = XY[idx][0]
        j = XY[idx][1]

        if ((done[idx] != 1) & (iter <= dicom_global_params.Ne * list_im_size[idx][1] * list_im_size[idx][0] - 1)):
            # First, decrypt for the lastest pixel
            # Diffusion
            temp = kC[idx][XY_P[idx][0]][XY_P[idx][1]]
            temp_str = xor(np.binary_repr(temp, width = dicom_global_params.NB_max), phi_dest_P[idx])
            kC[idx][XY_P[idx][0]][XY_P[idx][1]] = np.uint32(int(temp_str, 2))
            
            # Diffusion
            temp = kC[idx][i][j] # Current pixel after Permutations
            temp_str = xor(np.binary_repr(temp, width = dicom_global_params.NB_max), np.binary_repr(kC_minus[idx][0], width = dicom_global_params.NB_max)) # temp_value = I[i][j] XOR C[i-1][j]
            temp_str = xor(temp_str, phi_source_P[idx]) # temp_value XOR phi_source_P (result of chaotic map)
            kC[idx][i][j] = np.uint32(int(temp_str, 2))
            
            # Permutation
            temp = kC[idx][i][j]
            kC[idx][i][j] = kC[idx][XY_P[idx][0]][XY_P[idx][1]]
            kC[idx][XY_P[idx][0]][XY_P[idx][1]] = temp


            if ((i == 0) & (j == 0) & (list_n[idx] != 0)):
                # S-box for all pixels
                for height in range(list_im_size[idx][0]):
                    for width in range(list_im_size[idx][1]):
                        temp = kC[idx][height][width]
                        temp_str = sbox_subbyte(Sbox = Sbox, bit_sequence = np.binary_repr(temp, width = dicom_global_params.NB_max))
                        kC[idx][height][width] = np.uint32(int(temp_str, 2))


            # For the next pixel: Pass the diffused pixel's value to chaotic map
            kP_to_pcm[idx][0] = kC[idx][i][j]


            # Find kC_minus for the next pixel
            if (i > 0): # This means i - 1 is not out of range (i - 1 >= 0)
                if (j > 1): # This means k[i][j-2] is inside image because j - 2 is not out of range (j - 2 > 0)
                    kC_minus_next[idx][0] = np.uint32(kC[idx][i][j-2])
                else: # This means j - 2 is out of range (j - 2 < 0)
                    kC_minus_next[idx][0] = np.uint32(kC[idx][i-1][list_im_size[idx][1]-(2-j)]) # Go to the (N_k-(2-j))th pixel of the previous rows
            else: # This means i - 1 is out of range (i - 1 < 0)
                if (j > 1): # This means j - 2 is not out of range (j - 2 >= 0)
                    kC_minus_next[idx][0] = np.uint32(kC[idx][i][j-2])
                elif(j == 1): # This mean j - 2 == -1 and the next pixels is the first pixels of all the images
                    if (list_n[idx] > 0):
                        kC_minus_next[idx][0] = np.uint32(kC[idx][list_im_size[idx][0]-1][list_im_size[idx][1]-1])
                    else: # The last iteration
                        kC_minus_next[idx][0] = np.uint32(dicom_global_params.kC0[idx][0] & (2**list_num_bits_pre[idx] - 1))
                else: # This means j == 0
                    if (list_n[idx] > 0):
                        kC_minus_next[idx][0] = np.uint32(kC[idx][list_im_size[idx][0]-1][list_im_size[idx][1]-2])

            if (i > 0):
                if (j > 0):
                    XY[idx] = [i, j - 1]
                else:
                    XY[idx] = [i - 1, list_im_size[idx][1] - 1]
            else:
                if (j > 0):
                    XY[idx] = [i, j - 1]
                else:
                    if (list_n[idx] == 0):
                        XY[idx]   = [list_im_size[idx][0] - 1, list_im_size[idx][1] - 1]
                        done[idx] = 1
                    else:
                        XY[idx]     = [list_im_size[idx][0] - 1, list_im_size[idx][1] - 1]
                        list_n[idx] = list_n[idx] - 1
        else:
            kP_to_pcm[idx][0] = prev_kP_plus[idx][0]
            kC_minus_next[idx][0] = kC_minus[idx][0]

    return XY, kC_minus_next, kP_to_pcm, kC


# Function: MIE Permutation and Diffusion of Decryption processing for the kth multi-channel image
# k: Order of the image
# kC: Cipher image (just one image)
# XY: Current i and j (list [i, j])
# XY_P: New position [i', j']
# phi_source_Pk: After permutation, source plain pixel is XORed with phi_source_Pk
# phi_dest_Pk: After permutation, destination plain pixel is XORed with phi_dest_Pk
# c_minus: c- (just [[c[i][j-1]]])
# M_k: M_k
# N_k: N_k
# num_bits_pre: Number of bits to present a pixel
# n: Current iteration's order
# Return: 1. c_minus_next: c- for the next pixel
#         2. kP_to_pcm: Pixel's value used to impact chaotic map (the same type as kC_minus)
#         3. kC: kC
def dicom_mie_perm_diff_dec_mc_k(k, kC, XY_k, XY_Pk, phi_source_Pk, phi_dest_Pk, kC_minus_k, M_k, N_k, num_bits_pre_k, Sbox, n):
    # For the next pixel: Pass the diffused pixel's value to chaotic map
    kP_to_pcm = np.zeros_like(kC_minus_k, dtype=np.uint32)

    # Find kC_minus for the next pixel
    kC_minus_next = np.zeros_like(kC_minus_k, dtype=np.uint32)

    i = XY_k[0]
    j = XY_k[1]

    # First, decrypt for the lastest pixel
    # Diffusion
    temp = kC[XY_Pk[0]][XY_Pk[1]]
    temp_str = xor(np.binary_repr(temp, width = dicom_global_params.NB_max), phi_dest_Pk)
    kC[XY_Pk[0]][XY_Pk[1]] = np.uint32(int(temp_str, 2))
    
    # Diffusion
    temp = kC[i][j] # Current pixel after Permutations
    temp_str = xor(np.binary_repr(temp, width = dicom_global_params.NB_max), np.binary_repr(kC_minus_k[0], width = dicom_global_params.NB_max)) # temp_value = I[i][j] XOR C[i-1][j]
    temp_str = xor(temp_str, phi_source_Pk) # temp_value XOR phi_source_P (result of chaotic map)
    kC[i][j] = np.uint32(int(temp_str, 2))
    
    # Permutation
    temp = kC[i][j]
    kC[i][j] = kC[XY_Pk[0]][XY_Pk[1]]
    kC[XY_Pk[0]][XY_Pk[1]] = temp


    if ((i == 0) & (j == 0) & (n != 0)):
        # S-box for all pixels
        for height in range(M_k):
            for width in range(N_k):
                temp = kC[height][width]
                temp_str = sbox_subbyte(Sbox = Sbox, bit_sequence = np.binary_repr(temp, width = dicom_global_params.NB_max))
                kC[height][width] = np.uint32(int(temp_str, 2))


    # For the next pixel: Pass the diffused pixel's value to chaotic map
    kP_to_pcm[0] = kC[i][j]


    # Find kC_minus for the next pixel
    if (i > 0): # This means i - 1 is not out of range (i - 1 >= 0)
        if (j > 1): # This means k[i][j-2] is inside image because j - 2 is not out of range (j - 2 > 0)
            kC_minus_next[0] = np.uint32(kC[i][j-2])
        else: # This means j - 2 is out of range (j - 2 < 0)
            kC_minus_next[0] = np.uint32(kC[i-1][N_k-(2-j)]) # Go to the (N_k-(2-j))th pixel of the previous rows
    else: # This means i - 1 is out of range (i - 1 < 0)
        if (j > 1): # This means j - 2 is not out of range (j - 2 >= 0)
            kC_minus_next[0] = np.uint32(kC[i][j-2])
        elif(j == 1): # This mean j - 2 == -1 and the next pixels is the first pixels of all the images
            if (n > 0):
                kC_minus_next[0] = np.uint32(kC[M_k-1][N_k-1])
            else: # The last iteration
                kC_minus_next[0] = np.uint32(dicom_global_params.kC0[k][0] & (2**num_bits_pre_k - 1))
        else: # This means j == 0
            if (n > 0):
                kC_minus_next[0] = np.uint32(kC[M_k-1][N_k-2])
    
    if (i > 0):
        if (j > 0):
            XY_k = [i, j - 1]
        else:
            XY_k = [i - 1, N_k - 1]
    else:
        if (j > 0):
            XY_k = [i, j - 1]
        else:
            if (n == 0):
                XY_k = [M_k - 1, N_k - 1]
            else:
                XY_k = [M_k - 1, N_k - 1]

    return XY_k, kC_minus_next, kP_to_pcm, kC
