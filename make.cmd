nvcc -c cu_mat_test.cu
nvcc block_dim.obj cu_error_list.obj cu_mat_test.obj cu_mat.obj error_check.obj ode45.obj -lcublas -lcusolver -lcurand