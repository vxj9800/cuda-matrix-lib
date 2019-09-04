# Matrix class for CUDA
`cu_mat_test.cu` is the main file which will be compiled.  
`cu_mat.cu` is the source file.  
`cu_mat.hcu` is the cu_mat class header.  
`error_check.cu` and `cu_error_list.cu` contains supporting functions for error handelling.  
`block_dim.cu` is an attempt to automatically choose correct number of threads per 
block.
`imp_includes.hcu` is the header file that includes all the other important libraries 
and has some macro definitions for exception handling.

The files can be compiled with following command for g++  
`nvcc -c block_dim.cu cu_error_list.cu cu_mat.cu cu_mat_test.cu error_check.cu ode45.cu`  
`nvcc block_dim.o cu_error_list.o cu_mat.o cu_mat_test.o error_check.o ode45.o -lcublas -lcurand -lcusolver -o a.out`  

The files can be compiled with following command for Visual C++  
`nvcc -c block_dim.cu cu_error_list.cu cu_mat.cu cu_mat_test.cu error_check.cu ode45.cu`  
`nvcc block_dim.obj cu_error_list.obj cu_mat.obj cu_mat_test.obj error_check.obj ode45.obj -lcublas -lcurand -lcusolver`
