from nuqulib import Hamiltonian, element, nucl_ansatz
from qiskit_algorithms.minimum_eigensolvers import AdaptVQE, VQE
from qiskit_nature.second_q.circuit.library.ansatzes import UCC
from qiskit import QuantumCircuit
from qiskit import qpy
import numpy as np
import pickle
import argparse


def parse_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--file_path", default="/Users/ermal/workspace/NERSC/QC_Nuclear/tmp/tests/interaction_file/", \
                        type=str, help="path to files")
    parser.add_argument("--fn_NN", default="TwBME-HO_NN-only_N3LO_EM500_srg1.8_hw20_emax3_e2max6.kshell.snt", \
                        type=str, help="filename for NN interaction")
    parser.add_argument("--fn_3NF", default="ThBME_lnl_ms1_2_3.readable.txt", type=str, help="filename for NNN interaction")
    parser.add_argument("--ansatz", default='HF+Givens', type=str, help="ansatz to use: HF/HF+Givens/UCC/UCCSD/None")
    parser.add_argument("--params_file", default="", type=str, help="filename for params file for ansatz")
    parser.add_argument("--N", default=2, type=int, help="Number of neutrons")
    parser.add_argument("--Z", default=1, type=int, help="Number of protons")
    parser.add_argument("--ncsm", default=True, type=bool, help="whether to use no-core shell model")    
    parser.add_argument("--emax_truncate", default=3, type=int, help="emax truncate for single particle Hilbert space")
    parser.add_argument("--e3max", default=3, type=int, help="emax truncate for single particle Hilbert space for NNN interactions")
    #parser.add_argument("--jz", default=0, type=int, help="orbital angular momentum")
    parser.add_argument("--mapper", default="JordanWigner", type=str, help="Fermion to Qubit mapper")
    
    args = parser.parse_args([])

    return args


def main():
    args = parse_args()
    file_path = args.file_path
    fNN = args.fn_NN
    f3N = args.fn_3NF
    filename_NN = file_path+fNN
    if f3N == '':
        filename_NNN = None
    else:
        filename_NNN = file_path+f3N

    isotope_name = element[args.Z]+str(args.Z+args.N)
    print(isotope_name)
    mapperPath = file_path + isotope_name
    hamil = Hamiltonian(filename_NN, args.Z, args.N, ncsm=args.ncsm, verbose=False, emax_truncate=args.emax_truncate, \
                        e3max=args.e3max, fn_3NF=filename_NNN)
    
    H_1b_p, H_1b_n, H_jz_p, H_jz_n, H_pp, H_nn, H_pn, H_3b = hamil.mapping_opform(args.mapper,filepath=mapperPath)
    H_1b = H_1b_p + H_1b_n
    Hamil_NN = H_1b + H_pp + H_nn + H_pn
    if H_3b is not None:
        Hamil_NN = Hamil_NN + H_3b

    # Save the SparsePauliOp to a file
    with open('Hamiltonian_'+isotope_name+'_'+str(args.mapper)+'_emax_'+str(args.emax_truncate)+'_e3max_'+str(args.e3max)+'.pkl', 'wb') as f:
        pickle.dump(Hamil_NN, f)


    if args.ansatz!='None':
        if args.params_file!="":
            with open(args.params_file, 'rb') as f:
                params = pickle.load(f)
        else:
            params =None
            
        qc = nucl_ansatz(hamil.Hamildict, hamil.n_qubits, args.Z, args.N, params, method=args.ansatz, \
                         mapping_method=args.mapper, filepath=mapperPath, return_Gdict=False)
        
        with open('Ansatz_'+isotope_name+'_'+str(args.ansatz)+'_'+str(args.mapper)+'_emax_'+str(args.emax_truncate)+'_e3max_'+str(args.e3max)+'.qpy', 'wb') as f:
            qpy.dump(qc, f)

if __name__=='__main__':
    main()
