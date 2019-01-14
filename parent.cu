#include <iostream>
#include <ctime>
using namespace std;

#include "./cu_mat_lib/cu_matrix.cu"

int main()
{
    // clock_t begin = clock();

    look_for_errors;
    // cu_mat a = randn(2,3);
    cu_mat e = stepspace(0,5,1);
    //e.replace(2,4,3,4,{{10,11},{12,13},{14,15}});
    e = e.pow(2);
    e.get();
    report_errors;

    // clock_t end = clock();
    // double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    // cout << elapsed_secs;
    
    return (0);
}