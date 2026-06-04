Game of Life — Windows usage
=============================

This repository contains multiple implementations of Conway's Game of Life:
sequential, OpenMP, MPI+OpenMP, and CUDA.

Prerequisites (Windows)
- GCC/MinGW with OpenMP support (or Visual Studio for CUDA toolchain)
- MS-MPI (for the MPI version)
- CUDA Toolkit (for the CUDA version)

Build and run commands (PowerShell)

- Sequential
  - gcc code/life_sequential.c -o sequential -fopenmp
  - .\\sequential

- OpenMP
  - gcc code/life_openMP.c -o openmp -fopenmp
  - .\\openmp

- MPI + OpenMP
  - gcc -fopenmp life_openMP_MPI.c -o mixed.exe -I. -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x86" -lmsmpi
  - mpiexec -n 2 .\\mixed.exe

- CUDA
  - Example using Developper power shell for VS 22 as host compiler:
    nvcc -allow-unsupported-compiler -ccbin "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\<version>\\bin\\Hostx64\\x64\\cl.exe" code/life_cuda.cu -o life_cuda.exe
  - .\\life_cuda.exe

Notes
- Detailed compilation hints are present as comments at the bottom of each source file.
- This README is intentionally minimal and Windows-focused.
