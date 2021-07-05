# Benchmark results

## Speedup OpenFoam vs Ginkgo Backend

###  Results for Poisson equation only

The following figures show the speedup achieved by using ginkgo as backend compared to serial OpenFOAM execution. Here, only matrix solver which are available for both backends, i.e. CG and BiCGStab are compared against each other.
![figure](poisson-BiCGStabCGNoPrecond.png)

###  Results for Poisson and Momentum equation
![figure](momentum-BiCGStabCGNoPrecond.png)


## Speedup Unpreconditioned vs Preconditioned with Ginkgo Backend
The following figures show the impact of using BJ as preconditioner compared to non-preconditioned results.

###  Results for Poisson equation only
![figure](poisson-GKOBiCGStabCGBJ.png)

###  Results for Poisson and Momentum equation
![figure](momentum-GKOBiCGStabCGBJ.png)


## Speedup Unpreconditioned vs Preconditioned with OpenFOAM Backend
![figure](poisson-OFBiCGStabCGDIC.png)


## Speedup of Different Solver vs CG  with Ginkgo Backend
The following figures show the relative execution time of different ginkgo solvers compared to CG.

![figure](momentum-GKOSolver.png)

![figure](poisson-GKOSolver.png)

