import numpy as np
from mpi4py import MPI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class WaveEquation:
    def __init__(self, f, g, L, T, N, M, c):
        self.f = f
        self.g = g
        self.L = L
        self.T = T
        self.N = N
        self.M = M
        self.c = c
    def sequential_wave_equation(self):
        seq_dx = self.L / self.N
        seq_dt = self.T / self.M
        
        seq_x = np.linspace(0, self.L, self.N+1)
        seq_t = np.linspace(0, self.T, self.M+1)
        
        seq_phi = np.zeros((self.M+1, self.N+1))
        seq_phi[0, :] = self.f(seq_x)
        seq_phi[1, :] = self.f(seq_x) + seq_dt * self.g(seq_x)
        
        for k in range(1, self.M):
            for i in range(1, self.N):
                seq_phi[k+1, i] = self.c**2 * (seq_dt**2 / seq_dx**2) * (seq_phi[k, i+1] - 2 * seq_phi[k, i] + seq_phi[k, i-1]) + 2 * seq_phi[k, i] - seq_phi[k-1, i]
        
        return seq_x, seq_t, seq_phi
    def parallel_wave_equation(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        par_dx = self.L / self.N
        par_dt = self.T / self.M
        
        par_x_local = np.linspace(rank * (self.N // size) * par_dx, (rank + 1) * (self.N // size) * par_dx, self.N // size + 1)
        par_x = comm.gather(par_x_local, root=0)
        
        if rank == 0:
            par_x = np.concatenate(par_x)
            par_t = np.linspace(0, self.T, self.M+1)
            par_phi = np.zeros((self.M+1, self.N+1))
            par_phi[0, :] = self.f(par_x)
            par_phi[1, :] = self.f(par_x) + par_dt * self.g(par_x)
        else:
            par_t = None
            par_phi = None
        
        for k in range(1, self.M):
            par_phi_local = np.zeros_like(par_x_local)
            comm.Scatter(par_phi[k, :], par_phi_local, root=0)

            # Handle boundary values
            par_phi_local_new = np.zeros_like(par_phi_local)
            for i in range(1, len(par_x_local)-1):
                par_phi_local_new[i] = self.c**2 * (par_dt**2 / par_dx**2) * (par_phi_local[i+1] - 2 * par_phi_local[i] + par_phi_local[i-1]) + 2 * par_phi_local[i] - par_phi[k-1, i]

            # Gather the updated local phi values from all processes to process 0
            par_phi_new = comm.gather(par_phi_local_new, root=0)

            if rank == 0:
                par_phi_new_combined = np.concatenate(par_phi_new)
            else:
                par_phi_new_combined = None

            # Distribute the combined phi_new array to all processes
            par_phi_new_combined = comm.bcast(par_phi_new_combined, root=0)

            # Update phi for next time step
            if rank != 0:
                par_phi[k+1, 1:-1] = par_phi_new_combined[rank*(self.N // size):(rank + 1)*(self.N // size)]

            if rank == 0:
                par_phi[k+1, :] = par_phi_new_combined

        if rank == 0:
            return par_x, par_t, par_phi
        
    def plot_seq_and_par(self, seq_x, seq_t, seq_phi, par_x, par_t, par_phi):
        # Plotting
        if MPI.COMM_WORLD.Get_rank() == 0:
            plt.figure(figsize=(20, 10))
            plt.subplot(1, 4, 1)
            plt.plot(seq_x, seq_phi[-1, :], label = 'Sequential')
            plt.xlabel('x')
            plt.ylabel('\phi')
            plt.title('Final Time Step (Sequential)')
            plt.legend()

            plt.subplot(1, 4, 2)
            plt.plot(seq_t, seq_phi[:, self.N//2], label = 'Sequential')
            plt.xlabel('t')
            plt.ylabel('\phi')
            plt.title('Wave Propagation at x = L/2 (Sequential)')
            plt.legend()

            plt.subplot(1, 4, 3)
            plt.plot(par_x, par_phi[:, -1], label = 'Parallel')
            plt.xlabel('x')
            plt.ylabel('\phi')
            plt.title('Final Time Step (Parallel)')
            plt.legend()

            plt.subplot(1, 4, 4)
            plt.plot(par_t, par_phi[self.N//2, :], label = 'Parallel')
            plt.xlabel('t')
            plt.ylabel('\phi')
            plt.title('Wave Propagation at x = L/2 (Parallel)')
            plt.legend()

            plt.tight_layout()
            plt.savefig('wave_equation_plot.png')  # Save the plot to a file

def main():
    def f(x):
        return np.sin(np.pi * x)

    def g(x):
        return np.zeros_like(x)

    L = 1.0  # Interval Length
    T = 1.0  # Final time
    N = 100  # Number of spatial points
    M = 100  # Number of time steps
    c = 1.0  # Speed of light

    two_dimension = WaveEquation(f,g,L,T,N,M,c)
    # Sequential solution
    seq_x, seq_t, seq_phi = two_dimension.sequential_wave_equation()
    # Parallel solution
    par_x, par_t, par_phi = two_dimension.parallel_wave_equation()
    # Plotting 
    two_dimension.plot_seq_and_par(seq_x,seq_t,seq_phi,par_x,par_t,par_phi)

if __name__ == "__main__":
    main()
