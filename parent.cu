#include <iostream>
#include <ctime>
using namespace std;

#include "./cu_mat_lib/cu_matrix.cu"

int main()
{
    // clock_t begin = clock();

    look_for_errors;
    cu_mat a = ones(3,5);
    cu_mat b = randn(3,3);
    a.get(); b.get();
    a = b^2;
    a.get();
    // cu_mat a = {{1,2,3,4},{4,5,6}};
    // a.get();
    // cu_mat b = {{1,2},{4,5}};
    // cu_mat c = a*b;
    // a.get(); b.get(); c.get();

    // cu_mat a = randn(4,4);
    // cu_mat b = randn(4,1);
    //a.get(); b.get(); c.get();
    report_errors;

    // clock_t end = clock();
    // double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    // cout << elapsed_secs;
    
    return (0);
}