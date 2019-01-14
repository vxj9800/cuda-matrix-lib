#ifndef __ODE_SOLVER_DEFINED__
#define __ODE_SOLVER_DEFINED__

#include <functional>
#include <fstream>
#include <iomanip>
#include "cu_matrix.cu"

double eps(double n)
{
	double n_old;
	while ((1+n) != 1)
	{n_old = n; n/=2;}
	return n_old;
}

void ode45(std::function<cu_mat (cu_mat,cu_mat,cu_mat)> der, double t0, double tstep, double tf, cu_mat y0, cu_mat der_params)
{
	// Initiate Butcher tableau coefficients for dormand prince method.
	// Runge-kutta of order s can be written as k_s = h*f(t + c_s*h, y + a_s1*k_1 + a_s2*k_2 + a_s3*k_3 + ... + a_s(s-1)*k_(s-1))
	cu_mat c2 = 1/5, c3 = 3/10, c4 = 4/5, c5 = 8/9, c6 = 1, c7 = 1;
	cu_mat b11 = 35/384, 		b12 = 0, 			b13 = 500/1113, 	b14 = 125/192, 	b15 = −2187/6784, 	b16 = 11/84, 	b17 = 0;	// Used to calculate the fifth-order accurate solution
	cu_mat b21 = 5179/57600, 	b22 = 0, 			b23 = 7571/16695, 	b24 = 393/640, 	b25 = −92097/339200,b26 = 187/2100,	b27 = 1/40; // Used to calculate an alternative solution to compare with the fifth-order accurate solution
	cu_mat a71 = 35/384,		a72 = 0, 			a73 = 500/1113,		a74 = 125/192,	a75 = −2187/6784,	a76 = 11/84;
	cu_mat a61 = 9017/3168, 	a62 = −355/33, 		a63 = 46732/5247,	a64 = 49/176,	a65 = −5103/18656;
	cu_mat a51 = 19372/6561, 	a52 = −25360/2187,	a53 = 64448/6561,	a54 = −212/729;
	cu_mat a41 = 44/45, 		a42 = −56/15, 		a43 = 32/9;
	cu_mat a31 = 3/40, 			a32 = 9/40;
	cu_mat a21 = 1/5;

	// Define other required variables
	cu_mat t = stepspace(t0,tf,step), rel_tol = 0.001, abs_tol = 1e-6, hmax = (tf-t0)/10, h = tstep/1000.0;
	cu_mat k1, k2, k3, k4, k5, k6, k7, y1, z1, err, s;
	cu_mat t_start, t_end, h_min;
	size_t n_fail;

	// Print the initial conditions and time to the files
	cu_mat(t0).print("ode_time.txt"); y0.print("ode_data.txt")

	for(cu_mat loop = 1; loop < t.rows; ++loop)
	{
		t_start = t(loop,1); t_end = t(loop+1,1); hmin = 16*eps(t_start); n_fail = 0;
		while((t_start+h)<=t_end)
		{
			k1 = h*der(t_start,y0,der_params);
			k2 = h*der(t_start+c2*h,y0+a21*k1,der_params);
			k3 = h*der(t_start+c3*h,y0+a31*k1+a32*k2,der_params);
			k4 = h*der(t_start+c4*h,y0+a41*k1+a42*k2+a43*k3,der_params);
			k5 = h*der(t_start+c5*h,y0+a51*k1+a52*k2+a53*k3+a54*k4,der_params);
			k6 = h*der(t_start+c6*h,y0+a61*k1+a62*k2+a63*k3+a64*k4+a65*k5,der_params);
			k7 = h*der(t_start+c7*h,y0+a71*k1+a72*k2+a73*k3+a74*k4+a75*k5+a76*k6,der_params);	// Multiplication by a constant has to be implemented

			y1 = y0+b11*k1+b12*k2+b13*k3+b14*k4+b15*k5+b16*k6+b17*k7;
			z1 = y0+b21*k1+b22*k2+b23*k3+b24*k4+b25*k5+b26*k6+b27*k7;

			err = norm(z1-y1,inf);
			s = pow(rtol*h/2/err,1/5.0)

			if (err<atol)
			{
				t_start = t_start+h; y0 = y1;
				if (h*s<hmin) h = hmin;
				else if (h*s>hmax) h = hmax;
				else h = h*s;
				n_fail = 0;
				if (((h+t_start)>t_end) && (t_start != t_end)) h = t_end-t_start;
			}
			else
			{
				if (h*s*0.8<hmin) h = hmin;
				else if (h*s*0.8>hmax) h = hmax;
				else h = h*s*0.8;
				++n_fail;
			}
			confirm(n_fail<=100,"Errot: Error tolerances cannot be met.");
		}
	}
}

#endif