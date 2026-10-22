# 1-1_wave_equation
This paper presents the numerical solution of the 1+1 dimensional wave
equation using the finite differences and investigates its performance in both
sequential and parallel computing environments. The wave equation, derived
from Maxwell's equations under the Lorentz gauge and simplified to one spatial
dimension with zero charge density, is discretized using second-order central
differences in space and time. The initial and boundary value problem is solved
for arbitrary initial displacement and velocity functions. A parallel
implementation using MPI is developed to efficiently handle large-scale
simulations by distributing the spatial domain across multiple processes.
Numerical validation is performed by comparing the computed results with the
analytical solution for a sinusoidal initial condition, confirming the correctness
of the implementation and expected second-order convergence. Furthermore, a
performance study including speedup and efficiency plots demonstrates the
scalability of the parallel algorithm, with strong scaling behaviour analysed for
increasing numbers of processes. The results show that parallel computing
significantly reduces runtime while maintaining accuracy, making this approach
suitable for high-resolution wave simulations
