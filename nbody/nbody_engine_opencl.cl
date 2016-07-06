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

__kernel void ComputeBlockLocal( int offset_n1, int offset_n2,
                            __global const nbcoord_t* mass,
							__global const nbcoord_t* y,
							__global nbcoord_t* f, int yoff, int foff )
{
    int		b1 = get_global_id(0);
	int		n1 = b1 + offset_n1;
	__global const nbcoord_t*	rx = y + yoff;
	__global const nbcoord_t*	ry = rx + get_global_size(0);
	__global const nbcoord_t*	rz = rx + 2*get_global_size(0);
	__global const nbcoord_t*	vx = rx + 3*get_global_size(0);
	__global const nbcoord_t*	vy = rx + 4*get_global_size(0);
	__global const nbcoord_t*	vz = rx + 5*get_global_size(0);

	__global nbcoord_t*	frx = f + foff;
	__global nbcoord_t*	fry = frx + get_global_size(0);
	__global nbcoord_t*	frz = frx + 2*get_global_size(0);
	__global nbcoord_t*	fvx = frx + 3*get_global_size(0);
	__global nbcoord_t*	fvy = frx + 4*get_global_size(0);
	__global nbcoord_t*	fvz = frx + 5*get_global_size(0);

	nbcoord_t	x1 = rx[n1];
	nbcoord_t	y1 = ry[n1];
	nbcoord_t	z1 = rz[n1];

    nbcoord_t	res_x = 0.0;
	nbcoord_t	res_y = 0.0;
	nbcoord_t	res_z = 0.0;

	__local nbcoord_t	x2[NBODY_DATA_BLOCK_SIZE];
	__local nbcoord_t	y2[NBODY_DATA_BLOCK_SIZE];
	__local nbcoord_t	z2[NBODY_DATA_BLOCK_SIZE];
	__local nbcoord_t	m2[NBODY_DATA_BLOCK_SIZE];

	// NB! get_local_size(0) == NBODY_DATA_BLOCK_SIZE
	for( int b2 = 0; b2 < get_global_size(0); b2 += NBODY_DATA_BLOCK_SIZE )
	{
		int			n2 = b2 + offset_n2 + get_local_id(0);

		// Copy data block to local memory
		x2[ get_local_id(0) ] = rx[n2];
		y2[ get_local_id(0) ] = ry[n2];
		z2[ get_local_id(0) ] = rz[n2];
		m2[ get_local_id(0) ] = mass[n2];

		// Synchronize local work-items copy operations
		barrier( CLK_LOCAL_MEM_FENCE );

		nbcoord_t	local_res_x = 0.0;
		nbcoord_t	local_res_y = 0.0;
		nbcoord_t	local_res_z = 0.0;

		for( int n2 = 0; n2 != NBODY_DATA_BLOCK_SIZE; ++n2 )
		{
			nbcoord_t	dx = x1 - x2[n2];
			nbcoord_t	dy = y1 - y2[n2];
			nbcoord_t	dz = z1 - z2[n2];
			nbcoord_t	r2 = ( dx*dx + dy*dy + dz*dz );

			if( r2 < NBODY_MIN_R )
			    r2 = NBODY_MIN_R;

			nbcoord_t	r = sqrt( r2 );
			nbcoord_t	coeff = (m2[n2])/(r*r2);

			dx *= coeff;
			dy *= coeff;
			dz *= coeff;

			local_res_x -= dx;
			local_res_y -= dy;
			local_res_z -= dz;
		}

		// Synchronize local work-items computations
		barrier( CLK_LOCAL_MEM_FENCE );

		res_x += local_res_x;
		res_y += local_res_y;
		res_z += local_res_z;
	}

	frx[b1] = vx[b1];
	fry[b1] = vy[b1];
	frz[b1] = vz[b1];
	fvx[b1] = res_x;
	fvy[b1] = res_y;
	fvz[b1] = res_z;
}

//! a[i] += b[i]*c
__kernel void fmadd1( __global nbcoord_t* a, __global const nbcoord_t* b, nbcoord_t c )
{
	int		i = get_global_id(0);
	a[i] += b[i]*c;
}

//! a[i+aoff] = b[i+boff] + c[i+coff]*d
__kernel void fmadd2( __global nbcoord_t* a, __global const nbcoord_t* b, __global const nbcoord_t* c,
					  nbcoord_t d, int aoff, int boff, int coff )
{
	int		i = get_global_id(0);
	a[i+aoff] = b[i+boff] + c[i+coff]*d;
}

//! a[i+aoff] += sum( b[i+boff+k*bstride]*c[k], k=[0...csize) )
__kernel void fmaddn1( __global nbcoord_t* a, __global const nbcoord_t* b, constant nbcoord_t* c,
					  int bstride, int aoff, int boff, int csize )
{
	int			i = get_global_id(0);
	nbcoord_t	s = b[i+boff]*c[0];
	for( int k = 1; k < csize; ++k )
	{
		s += b[i+boff+k*bstride]*c[k];
	}
	a[i+aoff] += s;
}

//! a[i+aoff] = b[i+boff] + sum( c[i+coff+k*cstride]*d[k], k=[0...dsize) )
__kernel void fmaddn2( __global nbcoord_t* a, __global const nbcoord_t* b,__global const nbcoord_t* c, constant nbcoord_t* d,
					  int cstride, int aoff, int boff, int coff, int dsize )
{
	int			i = get_global_id(0);
	nbcoord_t	s = c[i+coff]*d[0];
	for( int k = 1; k < dsize; ++k )
	{
		s += c[i+coff+k*cstride]*d[k];
	}
	a[i+aoff] = b[i+boff] + s;
}

//! a[i+aoff] = sum( c[i+coff+k*cstride]*d[k], k=[0...dsize) )
__kernel void fmaddn3( __global nbcoord_t* a, __global const nbcoord_t* c, constant nbcoord_t* d,
					  int cstride, int aoff, int coff, int dsize )
{
	int			i = get_global_id(0);
	nbcoord_t	s = c[i+coff]*d[0];
	for( int k = 1; k < dsize; ++k )
	{
		s += c[i+coff+k*cstride]*d[k];
	}
	a[i+aoff] = s;
}

//! *result = max( fabs(a[k]), k=[0...asize) )
__kernel void fmabsmax( __global const nbcoord_t* a, __global nbcoord_t* result )
{
}

