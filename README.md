# Matrix class for CUDA
`parent.cu` is the main file which will be compiled.  
`cu_matrix.cu` is the include file.  
`cu_matrix_class.cu` contains cu_matrix class.  
`cu_matrix_functions.cu` contains supported extra functions (randn for now).  
`error_check.cu` has supporting functions for error handelling.  

`parent.cu` can be compiled with following command for visual studio.  
`nvcc parent.cu -lcublas -lcurand -lcusolver`

`parent.cu` can be compiled with following command for g++.  
`nvcc parent.cu -std=c++14 -lcublas -lcurand -lcusolver`