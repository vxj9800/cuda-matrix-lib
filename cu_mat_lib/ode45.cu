#ifndef __ODE_SOLVER_DEFINED__
#define __ODE_SOLVER_DEFINED__

#include <functional>
#include <fstream>
#include <iomanip>

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
	cu_mat c2 = 1/5.0, c3 = 3/10.0, c4 = 4/5.0, c5 = 8/9.0, c6 = 1.0, c7 = 1.0;
	cu_mat b11 = 35/384.0, b12 = 0.0, b13 = 500/1113.0, b14 = 125/192.0, b15 = -2187/6784.0, b16 = 11/84.0, b17 = 0.0;
	cu_mat b21 = 5179/57600.0, b22 = 0.0, b23 = 7571/16695.0, b24 = 393/640.0, b25 = -92097/339200.0, b26 = 187/2100.0, b27 = 1/40.0;
	cu_mat a71 = 35/384.0, a72 = 0.0, a73 = 500/1113.0, a74 = 125/192.0, a75 = -2187/6784.0, a76 = 11/84.0;
	cu_mat a61 = 9017/3168.0, a62 = -355/33.0, a63 = 46732/5247.0, a64 = 49/176.0, a65 = -5103/18656.0;
	cu_mat a51 = 19372/6561.0, a52 = -25360/2187.0, a53 = 64448/6561.0, a54 = -212/729.0;
	cu_mat a41 = 44/45.0, a42 = -56/15.0, a43 = 32/9.0;
	cu_mat a31 = 3/40.0, a32 = 9/40.0;
	cu_mat a21 = 1/5.0;

	// Define other required variables
	cu_mat t = stepspace(t0,tf,tstep), rtol = 0.001, atol = 1e-6, hmax = 0, h = tstep/1000.0;
	cu_mat k1 = y0, k2 = y0, k3 = y0, k4 = y0, k5 = y0, k6 = y0, k7 = y0, y1 = y0, z1 = y0, err = 0, s = 0;
	cu_mat t_start = 0, t_end = 0, hmin = 0, sf = 0.8;
	size_t n_fail = 0;

	// Print the initial conditions and time to the files
	ofstream sim_time, sim_data;
	sim_time.open("ode_time.txt",ios::out | ios::trunc);
	sim_data.open("ode_data.txt",ios::out | ios::trunc);
	cu_mat(t).print(sim_time); y0.print(sim_data);

	for(int loop = 1; loop < t.rows(); ++loop)
	{
		t_start = t(loop,1); t_end = t(loop+1,1); h = tstep/1000.0;
		hmax = double(t_end-t_start)/10.0; hmin = 16.0*eps(double(t_start)); n_fail = 0;
		while(double((t_start+h)<=t_end))
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

			err = norm(z1-y1,INFINITY);
			// err.print("ode_err.txt",0); s.print("ode_s.txt",0); h.print("ode_h.txt",0);
			if (double(err==0))
			{
				s = 2;
			}
			else
			{
				s = pow(double(rtol*h/2.0/err),1/5.0);
			}
			// s.get();
			if (double(err<atol))
			{
				t_start = t_start+h; y0 = y1;
				// (h*s).get(); ((h*s)<hmin).get(); ((h*s)>hmax).get();
				if (double((h*s)<hmin)){h = hmin; (h*s).get();}
				else if (double((h*s)>hmax)){h = hmax;}
				else {h = h*s;}
				n_fail = 0;
				if (double(((h+t_start)>t_end) && (t_start != t_end))) {h = t_end-t_start;}
			}
			else
			{
				//cout << "why it came here?" << endl;
				if (double((h*s*sf)<hmin)) h = hmin;
				else if (double((h*s*sf)>hmax)) h = hmax;
				else h = h*s*sf;
				++n_fail;
			}
			confirm(n_fail<=100,"Errot: Error tolerances cannot be met.");
			// return;
		}
		y0.print(sim_data);
	}
	sim_time.close(); sim_data.close();
}

#endif