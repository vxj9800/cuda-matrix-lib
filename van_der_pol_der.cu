cu_mat der(cu_mat &t, cu_mat &x, cu_mat &params)
{
    // cout << x.rows() << " " << x.cols() << endl;
    cu_mat dx = {{x(2)},{params(1)*(cu_mat(1)-x(1).pow(2))*x(2)-x(1)}};
	return dx;
}