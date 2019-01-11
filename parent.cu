#include <iostream>
#include <ctime>
using namespace std;

#include "./cu_mat_lib/cu_matrix.cu"

int main()
{
    // clock_t begin = clock();

    look_for_errors;
    cu_mat a = randn(2,3);
    cu_mat b = randn(2,5);
    cu_mat c = randn(4,5);
    cu_mat d = randn(4,3);
    cu_mat e = 5;
    a.get(); b.get(); c.get(); d.get(); e.get();
    e = {{a,b},{c,d}};
    e.get();
    report_errors;

    // clock_t end = clock();
    // double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    // cout << elapsed_secs;
    
    return (0);
}