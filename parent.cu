#include <iostream>
#include <ctime>
using namespace std;

#include "./cu_mat_lib/cu_matrix.cu"
#include "./cu_mat_lib/ode45.cu"
#include "der.cu"

int main()
{
    clock_t begin = clock();
    cudaDeviceReset();
    look_for_errors;
    // cu_mat a = randn(2,3);
    // cu_mat e = 10;
    // if(e<(cu_mat(INFINITY)*cu_mat(10.0)))
    // {
    //     cout << "it worked.";
    // }
    // else{cout << "it did not work.";}
    cu_mat x0 = 0, params = 0;
    ode45(der,0,0.001,5,x0,params);

    //e.replace(2,4,3,4,{{10,11},{12,13},{14,15}});
    report_errors;

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << elapsed_secs;
    
    return (0);
}