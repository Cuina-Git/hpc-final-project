README

1.  Dir declaration:
    Explicit_Euler: use explicit Euler method to slove the problem
    Implicit_Euler: use implicit Euler method to slove the problem
    Code_verification: Verify the exact and approximate solutions

2.  Compile:
    make main.out -> main.out (executable file)
    PS: PETSC_DIR in Makefile is the path where PETSC be installed. 

3.  Run:
    bsub<ty_script -> log.out result.log

4.  Parameters:
    -m "Number of space grids"
    -n "Number of time grids"
