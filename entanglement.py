from datetime import datetime
import h5py,os,sys
# Set up threading options for parallel solver
os.environ['OMP_NUM_THREADS']= str(int(1)) # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']= str(int(1)) # set number of MKL threads to run in parallel
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from multiprocessing import Pool,freeze_support
from psutil import cpu_count
from functools import partial

L = int(sys.argv[1])
reps = 100
dis = [int(sys.argv[3])]
chimax = int(sys.argv[2])

psi0 = qtn.tensor_gen.MPS_neel_state(L)
psi0.show()

ts = np.logspace(np.log10(0.05),np.log10(500),151,endpoint=True,base=10)
# dt = 0.05
dt = ts[1]-ts[0]
print('Timestep = ', dt)

def mutinf_subsys(psi, sysa, sysb):
        """Calculate the mutual information of two subsystems of a pure state,
            possibly using an approximate lanczos method for large subsytems.
                """

        print(sysa,sysb)
        rho_ab = psi.partial_trace_compress(sysa, sysb)
        rho_ab_lo = rho_ab.aslinearoperator(['kA','kB'],['bA','bB'])
        hab = qu.calc.entropy(rho_ab_lo.to_dense())

        rho_a = psi.partial_trace(sysa)
        rho_a_lo = rho_a.aslinearoperator(rho_a.upper_inds,rho_a.lower_inds)
        ha = qu.calc.entropy(rho_a_lo.to_dense())

        rho_b = psi.partial_trace(sysb)
        rho_b_lo = rho_b.aslinearoperator(rho_b.upper_inds,rho_b.lower_inds)
        hb = qu.calc.entropy(rho_b_lo.to_dense())

        return hb + ha - hab

def run(d,p):

        builder = qtn.SpinHam1D(S=1/2)
        builder += 0.5, '+', '-'
        builder += 0.5, '-', '+'
        builder += 1.0, 'Z', 'Z'
        np.random.seed()
        dlist = np.zeros(L)
        for i in range(L):
            dlist[i] = np.random.uniform(-d,d)
            builder[i] += dlist[i], 'Z'
        H = builder.build_local_ham(L)
        H_MPO = builder.build_mpo(L)

        tebd = qtn.TEBD(psi0,H,progbar=False)

        tebd.split_opts['cutoff'] = 1e-10
        tebd.split_opts['max_bond'] = chimax

        mz_t_j = []
        be_t_b = []
        sg_t_b = [] 
        energy = []

        tstep = 0
        neg = np.zeros((L,len(ts)))
        mut_inf = np.zeros((L,len(ts)))
        for psit in tebd.at_times(ts, dt=dt):
            mz_j = []
            be_b = []
            sg_b = []

            mz_j += [psit.magnetization(0)]

            for j in range(1, L):
                mz_j += [psit.magnetization(j, cur_orthog=j - 1)]
                be_b += [psit.entropy(j, cur_orthog=j)]
                sg_b += [psit.schmidt_gap(j, cur_orthog=j)]

            block_size_list = [i for i in range(L//2-3)]

            for block_size in block_size_list:

                temp = []
                temp2 = []
                # EVEN SPACINGS
                temp += [psit.logneg_subsys([j for j in range(L//2-block_size)],[j for j in range(L//2+block_size,L)])]
                temp2 += [mutinf_subsys(psit,[j for j in range(L//2-block_size)],[j for j in range(L//2+block_size,L)])]
                if len(temp)>0:
                    neg[2*block_size,tstep] += np.mean(temp)
                    mut_inf[2*block_size,tstep] += np.mean(temp2)

                temp = []
                temp2 = []
                # ODD SPACINGS
                temp += [psit.logneg_subsys([j for j in range(L//2-block_size-1)],[j for j in range(L//2+block_size,L)])]
                temp += [psit.logneg_subsys([j for j in range(L//2-block_size)],[j for j in range(L//2+block_size+1,L)])]
                temp2 += [mutinf_subsys(psit,[j for j in range(L//2-block_size-1)],[j for j in range(L//2+block_size,L)])]
                temp2 += [mutinf_subsys(psit,[j for j in range(L//2-block_size)],[j for j in range(L//2+block_size+1,L)])]
                if len(temp)>0:
                    neg[2*block_size+1,tstep] += np.mean(temp)
                    mut_inf[2*block_size+1,tstep] += np.mean(temp2)

            mz_t_j += [mz_j]
            be_t_b += [be_b]
            sg_t_b += [sg_b]
            energy += [qtn.expec_TN_1D(tebd.pt.H,H_MPO,tebd.pt)]

            tstep += 1

        tebd.pt.show()
        #print(tebd.err)
        #print("Initial energy:", qtn.expec_TN_1D(psi0.H, H_MPO, psi0))
        #print("Final energy:", qtn.expec_TN_1D(tebd.pt.H , H_MPO, tebd.pt))
        #print('TYPE',type(tebd.pt))
        #print('TYPE',type(np.array(tebd.pt)))

        if not os.path.exists('quimb'):
            os.makedirs('quimb/')
        with h5py.File('quimb/dyn_L%s_X%s_d%.2f_p%.2f.h5' %(L,chimax,d,p),'w') as hf:
            hf.create_dataset('mag',data=np.array(mz_t_j))
            hf.create_dataset('ent',data=np.array(be_t_b))
            hf.create_dataset('schmidt',data=np.array(sg_t_b))
            hf.create_dataset('neg',data=np.array(neg))
            hf.create_dataset('mut_inf',data=np.array(mut_inf))
            hf.create_dataset('err',data=[tebd.err])
            hf.create_dataset('energy',data=energy)
            hf.create_dataset('times',data=ts)
            hf.create_dataset('dlist',data=dlist)
            hf.create_dataset('state',data=str(tebd.pt),dtype=h5py.string_dtype(encoding='ascii'))

#----------------------------------------------------------------
if __name__ == '__main__': 

    startTime = datetime.now()
    print('Start time: ', startTime)
    freeze_support()

    pool = Pool(processes = cpu_count())
    for d in dis:
        plist = range(reps)
        pool.map(partial(run,d),plist)
    
    pool.close()
    pool.join()

    print('End time: ',datetime.now()-startTime)
