#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// Dense fcompute with local memory load (each particle interacts with each).
__kernel void ComputeBlockLocal(int offset_n1, int offset_n2,
								__global const nbcoord_t* mass,
								__global const nbcoord_t* y,
								__global nbcoord_t* f, int yoff,
								int foff, int points_count, int stride)
{
	int		n1 = get_global_id(0) + offset_n1;
	__global const nbcoord_t*	rx = y + yoff;
	__global const nbcoord_t*	ry = rx + stride;
	__global const nbcoord_t*	rz = rx + 2 * stride;
	__global const nbcoord_t*	vx = rx + 3 * stride;
	__global const nbcoord_t*	vy = rx + 4 * stride;
	__global const nbcoord_t*	vz = rx + 5 * stride;

	__global nbcoord_t*	frx = f + foff;
	__global nbcoord_t*	fry = frx + stride;
	__global nbcoord_t*	frz = frx + 2 * stride;
	__global nbcoord_t*	fvx = frx + 3 * stride;
	__global nbcoord_t*	fvy = frx + 4 * stride;
	__global nbcoord_t*	fvz = frx + 5 * stride;

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
	for(int b2 = 0; b2 < points_count; b2 += NBODY_DATA_BLOCK_SIZE)
	{
		int			n2 = b2 + offset_n2 + get_local_id(0);

		// Copy data block to local memory
		x2[ get_local_id(0) ] = rx[n2];
		y2[ get_local_id(0) ] = ry[n2];
		z2[ get_local_id(0) ] = rz[n2];
		m2[ get_local_id(0) ] = mass[n2];

		// Synchronize local work-items copy operations
		barrier(CLK_LOCAL_MEM_FENCE);

		nbcoord_t	local_res_x = 0.0;
		nbcoord_t	local_res_y = 0.0;
		nbcoord_t	local_res_z = 0.0;

		for(int n2 = 0; n2 != NBODY_DATA_BLOCK_SIZE; ++n2)
		{
			nbcoord_t	dx = x1 - x2[n2];
			nbcoord_t	dy = y1 - y2[n2];
			nbcoord_t	dz = z1 - z2[n2];
			nbcoord_t	r2 = (dx * dx + dy * dy + dz * dz);

			if(r2 < NBODY_MIN_R)
			{
				r2 = NBODY_MIN_R;
			}

			nbcoord_t	r = sqrt(r2);
			nbcoord_t	coeff = (m2[n2]) / (r * r2);

			dx *= coeff;
			dy *= coeff;
			dz *= coeff;

			local_res_x -= dx;
			local_res_y -= dy;
			local_res_z -= dz;
		}

		// Synchronize local work-items computations
		barrier(CLK_LOCAL_MEM_FENCE);

		res_x += local_res_x;
		res_y += local_res_y;
		res_z += local_res_z;
	}

	frx[n1] = vx[n1];
	fry[n1] = vy[n1];
	frz[n1] = vz[n1];
	fvx[n1] = res_x;
	fvy[n1] = res_y;
	fvz[n1] = res_z;
}

#define MAX_STACK_SIZE 64
#define NULL_NODE_PTR (-1)
#define distance_to_node_radius_ratio 10
struct node_bh
{
	nbcoord_t x;
	nbcoord_t y;
	nbcoord_t z;
	nbcoord_t m;
	nbcoord_t r2;
	int left;
	int right;
};

// Sparse fcompute using Kd-tree traverse (Barnes-Hut engine)
__kernel void ComputeTreeBH(int offset_n1, int offset_n2,
							__global const nbcoord_t* mass,
							__global const nbcoord_t* y,
							__global nbcoord_t* f, int yoff,
							int foff, int points_count, int stride,
							__global struct node_bh* tree)
{
	int		n1 = get_global_id(0) + offset_n1;
	__global const nbcoord_t*	rx = y + yoff;
	__global const nbcoord_t*	ry = rx + stride;
	__global const nbcoord_t*	rz = rx + 2 * stride;
	__global const nbcoord_t*	vx = rx + 3 * stride;
	__global const nbcoord_t*	vy = rx + 4 * stride;
	__global const nbcoord_t*	vz = rx + 5 * stride;

	__global nbcoord_t*	frx = f + foff;
	__global nbcoord_t*	fry = frx + stride;
	__global nbcoord_t*	frz = frx + 2 * stride;
	__global nbcoord_t*	fvx = frx + 3 * stride;
	__global nbcoord_t*	fvy = frx + 4 * stride;
	__global nbcoord_t*	fvz = frx + 5 * stride;

	nbcoord_t	x1 = rx[n1];
	nbcoord_t	y1 = ry[n1];
	nbcoord_t	z1 = rz[n1];

	nbcoord_t	res_x = 0.0;
	nbcoord_t	res_y = 0.0;
	nbcoord_t	res_z = 0.0;

	struct node_bh	stack_data[MAX_STACK_SIZE] = {};
	int	stack = 0;
	int	stack_head = stack;

	stack_data[stack++] = tree[0];
	while(stack != stack_head)
	{
		struct node_bh	curr = stack_data[--stack];
		nbcoord_t		dx = x1 - curr.x;
		nbcoord_t		dy = y1 - curr.y;
		nbcoord_t		dz = z1 - curr.z;
		nbcoord_t		r2 = (dx * dx + dy * dy + dz * dz);

		if(r2 > distance_to_node_radius_ratio * curr.r2)
		{
			if(r2 < NBODY_MIN_R)
			{
				r2 = NBODY_MIN_R;
			}

			nbcoord_t	r = sqrt(r2);
			nbcoord_t	coeff = curr.m / (r * r2);

			dx *= coeff;
			dy *= coeff;
			dz *= coeff;
			res_x -= dx;
			res_y -= dy;
			res_z -= dz;
		}
		else
		{
			if(curr.left != NULL_NODE_PTR)
			{
				stack_data[stack++] = tree[curr.left];
			}
			if(curr.right != NULL_NODE_PTR)
			{
				stack_data[stack++] = tree[curr.right];
			}
		}
	}

	frx[n1] = vx[n1];
	fry[n1] = vy[n1];
	frz[n1] = vz[n1];
	fvx[n1] = res_x;
	fvy[n1] = res_y;
	fvz[n1] = res_z;
}

//! a[i] = value
__kernel void fill(__global nbcoord_t* a, nbcoord_t value)
{
	int		i = get_global_id(0);
	if(i < get_global_size(0))
	{
		a[i] = value;
	}
}

//! a[i] += b[i]*c
__kernel void fmadd1(int offset, __global nbcoord_t* a, __global const nbcoord_t* b, nbcoord_t c)
{
	int		i = get_global_id(0) + offset;
	a[i] += b[i] * c;
}

//! a[i+aoff] = b[i+boff] + c[i+coff]*d
__kernel void fmadd2(__global nbcoord_t* a, __global const nbcoord_t* b, __global const nbcoord_t* c,
					 nbcoord_t d, int aoff, int boff, int coff)
{
	int		i = get_global_id(0);
	a[i + aoff] = b[i + boff] + c[i + coff] * d;
}

//! *result = max( fabs(a[k]), k=[0...asize) )
__kernel void fmaxabs(__global const nbcoord_t* a, int aoff, int alast, __global nbcoord_t* result, int roff)
{
	int			i = get_global_id(0);
	int			first = i * NBODY_DATA_BLOCK_SIZE + aoff;

	if(first >= alast)
	{
		return;
	}

	nbcoord_t	r = fabs(a[first]);

	for(int n = 0; n != NBODY_DATA_BLOCK_SIZE; ++n)
	{
		int aindex = first + n;
		if(aindex < alast)
		{
			r = max(r, fabs(a[aindex]));
		}
	}

	result[i + roff] = r;
}

