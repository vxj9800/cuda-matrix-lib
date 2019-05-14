#include <iostream>
#include <ctime>
using namespace std;

#include "./cu_mat_lib/cu_matrix.cu"
#include "./cu_mat_lib/ode45.cu"
#include "der.cu"
// #include "van_der_pol_der.cu"

int main()
{
    clock_t begin = clock();
    cudaDeviceReset();
    look_for_errors;
    cu_mat a = randn(3,3);
    cu_mat e = 5;
    //e = a(2,2);
    //a.get();
    cu_mat f = (a^0)*e;
    cout << f.rows() << endl << f.cols() << endl;
    f.get();
    // cu_mat x0 = 0, params = 0;
    // ode45(der,0,0.001,5,x0,params);
    // cu_mat x0 = {{2},{0}}, params = 1000;
    // ode45(der,0,50,3000,x0,params);
    report_errors;
    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << elapsed_secs;
    
    return (0);
}