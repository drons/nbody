#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void ComputeBlock( int offset_n1, int offset_n2,
                            __global const nbcoord_t* vert_x,
							__global const nbcoord_t* vert_y,
							__global const nbcoord_t* vert_z,
							__global const nbcoord_t* mass,
							__global nbcoord_t* dv_x,
							__global nbcoord_t* dv_y,
							__global nbcoord_t* dv_z )
{
    int			b1 = get_global_id(0);
	int			n1 = b1 + offset_n1;
	nbcoord_t	x1 = vert_x[n1];
	nbcoord_t	y1 = vert_y[n1];
	nbcoord_t	z1 = vert_z[n1];

    nbcoord_t	res_x = 0.0;
	nbcoord_t	res_y = 0.0;
	nbcoord_t	res_z = 0.0;

	for( int b2 = 0; b2 != get_global_size(0); ++b2 )
	{
		int			n2 = b2 + offset_n2;
		nbcoord_t	dx = x1 - vert_x[n2];
		nbcoord_t	dy = y1 - vert_y[n2];
		nbcoord_t	dz = z1 - vert_z[n2];
		nbcoord_t	r2 = ( dx*dx + dy*dy + dz*dz );

        if( r2 < NBODY_MIN_R )
		    r2 = NBODY_MIN_R;

        nbcoord_t	r = sqrt( r2 );
		nbcoord_t	coeff = (mass[n2])/(r*r2);

		dx *= coeff;
		dy *= coeff;
		dz *= coeff;

        res_x -= dx;
		res_y -= dy;
		res_z -= dz;
	}
	dv_x[b1] = res_x;
	dv_y[b1] = res_y;
	dv_z[b1] = res_z;
}

