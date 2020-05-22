/*
 * cu_mat_test.cu
 *
 *  Created on: Jun 6, 2019
 *      Author: vatsal
 */

#include "cu_mat.hcu"
#include <ctime>

#define numEquations 3200000

cu_mat der(const cu_mat &t, const cu_mat &x, const cu_mat &params)
{
	return std::move(cos(t*ones(numEquations,1)));
}

int main()
{
	clock_t begin = clock();
	look_for_errors;

//	cu_mat a = {{1,1,1},{0,2,5},{2,5,-1}};
//	cu_mat b = {{6},{-4},{27}};
//	cu_mat c = mld(cu_mat({{1,1,1},{0,2,5},{2,5,-1}}),b);
//	std::cout << (cu_mat(3.0)+a).is_rvalue() << std::endl;
//	b.get(); c.get();
//	cu_mat A = randn(5), b = randn(5,1);
//	cu_mat c = mld(A,b);
	cu_mat x0 = zeros(numEquations,1), params = 0;
	ode45(der,0,5,x0,params);
	// static cu_mat t = stepspace(0,5,0.001);
	// static cu_mat t_start = 0;
	// t_start = t(1,1);
	// size_t n_ele = 5001, tc = block_dim(n_ele);
	// std::cout << "Threads per block: " << tc << std::endl;

	report_errors;
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << elapsed_secs << std::endl;
//	cudaDeviceReset();
	return 0;
}
