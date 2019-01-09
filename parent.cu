#include <iostream>
#include <ctime>
using namespace std;

#include "./cu_mat_lib/cu_matrix.cu"

int main()
{
    // clock_t begin = clock();

    look_for_errors;
    cu_mat a = randn(5,10);
    cu_mat b = randn(3,2);
    cu_mat c = vertcat(a(2,2,6,8),b);
    a.get(); a(2,3,6,7).get(); b.get(); c.get();
    report_errors;

    // clock_t end = clock();
    // double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    // cout << elapsed_secs;
    
    return (0);
}