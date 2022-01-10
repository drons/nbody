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
		int	n2 = b2 + offset_n2 + get_local_id(0);

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

		for(int local_n2 = 0; local_n2 != NBODY_DATA_BLOCK_SIZE; ++local_n2)
		{
			nbcoord_t	dx = x1 - x2[local_n2];
			nbcoord_t	dy = y1 - y2[local_n2];
			nbcoord_t	dz = z1 - z2[local_n2];
			nbcoord_t	r2 = (dx * dx + dy * dy + dz * dz);

			if(r2 < NBODY_MIN_R)
			{
				r2 = NBODY_MIN_R;
			}

			nbcoord_t	r = sqrt(r2);
			nbcoord_t	coeff = (m2[local_n2]) / (r * r2);

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

// Sparse fcompute using Kd-tree traverse (Barnes-Hut engine)
// Cycle body access by index
// Tree traverse with stack
__kernel void ComputeTreeBH(int offset_n1, int points_count, int tree_size,
							__global const nbcoord_t* y,
							__global nbcoord_t* f,
							__global const nbcoord_t* tree_cmx,
							__global const nbcoord_t* tree_cmy,
							__global const nbcoord_t* tree_cmz,
							__global const nbcoord_t* tree_mass,
							__global const nbcoord_t* tree_crit_r2)
{
	int		n1 = get_global_id(0) + offset_n1;
	int		stride = points_count;
	__global const nbcoord_t*	rx = y;
	__global const nbcoord_t*	ry = rx + stride;
	__global const nbcoord_t*	rz = rx + 2 * stride;
	__global const nbcoord_t*	vx = rx + 3 * stride;
	__global const nbcoord_t*	vy = rx + 4 * stride;
	__global const nbcoord_t*	vz = rx + 5 * stride;

	__global nbcoord_t*	frx = f;
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

	int stack_data[MAX_STACK_SIZE] = {};
	int	stack = 0;
	int	stack_head = stack;

	stack_data[stack++] = NBODY_HEAP_ROOT_INDEX;
	while(stack != stack_head)
	{
		int			curr = stack_data[--stack];
		nbcoord_t	dx = x1 - tree_cmx[curr];
		nbcoord_t	dy = y1 - tree_cmy[curr];
		nbcoord_t	dz = z1 - tree_cmz[curr];
		nbcoord_t	r2 = (dx * dx + dy * dy + dz * dz);

		if(r2 > tree_crit_r2[curr])
		{
			if(r2 < NBODY_MIN_R)
			{
				r2 = NBODY_MIN_R;
			}

			nbcoord_t	r = sqrt(r2);
			nbcoord_t	coeff = tree_mass[curr] / (r * r2);

			dx *= coeff;
			dy *= coeff;
			dz *= coeff;
			res_x -= dx;
			res_y -= dy;
			res_z -= dz;
		}
		else
		{
			int	left = left_idx(curr);
			int	rght = rght_idx(curr);
			if(left < tree_size)
			{
				stack_data[stack++] = left;
			}
			if(rght < tree_size)
			{
				stack_data[stack++] = rght;
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

// Sparse fcompute using Kd-tree traverse (Barnes-Hut engine)
// Cycle body access by index
// Stackless tree traverse
__kernel void ComputeTreeBHSL(int offset_n1, int points_count, int tree_size,
							  __global const nbcoord_t* y,
							  __global nbcoord_t* f,
							  __global const nbcoord_t* tree_cmx,
							  __global const nbcoord_t* tree_cmy,
							  __global const nbcoord_t* tree_cmz,
							  __global const nbcoord_t* tree_mass,
							  __global const nbcoord_t* tree_crit_r2)
{
	int		n1 = get_global_id(0) + offset_n1;
	int		stride = points_count;
	__global const nbcoord_t*	rx = y;
	__global const nbcoord_t*	ry = rx + stride;
	__global const nbcoord_t*	rz = rx + 2 * stride;
	__global const nbcoord_t*	vx = rx + 3 * stride;
	__global const nbcoord_t*	vy = rx + 4 * stride;
	__global const nbcoord_t*	vz = rx + 5 * stride;

	__global nbcoord_t*	frx = f;
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

	int	curr = NBODY_HEAP_ROOT_INDEX;
	do
	{
		nbcoord_t	dx = x1 - tree_cmx[curr];
		nbcoord_t	dy = y1 - tree_cmy[curr];
		nbcoord_t	dz = z1 - tree_cmz[curr];
		nbcoord_t	r2 = (dx * dx + dy * dy + dz * dz);

		if(r2 > tree_crit_r2[curr])
		{
			if(r2 < NBODY_MIN_R)
			{
				r2 = NBODY_MIN_R;
			}

			nbcoord_t	r = sqrt(r2);
			nbcoord_t	coeff = tree_mass[curr] / (r * r2);

			dx *= coeff;
			dy *= coeff;
			dz *= coeff;
			res_x -= dx;
			res_y -= dy;
			res_z -= dz;
			curr = next_down(curr);
		}
		else
		{
			curr = next_up(curr, tree_size);
		}
	}
	while(curr != NBODY_HEAP_ROOT_INDEX);

	frx[n1] = vx[n1];
	fry[n1] = vy[n1];
	frz[n1] = vz[n1];
	fvx[n1] = res_x;
	fvy[n1] = res_y;
	fvz[n1] = res_z;
}

// Sparse fcompute using Kd-tree traverse (Barnes-Hut engine)
// Traverse starts form a tree node
// Tree traverse with stack
__kernel void ComputeHeapBH(int offset_n1, int points_count, int tree_size,
							__global const nbcoord_t* y,
							__global nbcoord_t* f,
							__global const nbcoord_t* tree_cmx,
							__global const nbcoord_t* tree_cmy,
							__global const nbcoord_t* tree_cmz,
							__global const nbcoord_t* tree_mass,
							__global const nbcoord_t* tree_crit_r2,
							__global const int* body_n)
{
	int		tree_offset = points_count - 1 + NBODY_HEAP_ROOT_INDEX;
	int		stride = points_count;
	int		tn1 = get_global_id(0) + offset_n1 + tree_offset;

	int			n1 = body_n[tn1];
	nbcoord_t	x1 = tree_cmx[tn1];
	nbcoord_t	y1 = tree_cmy[tn1];
	nbcoord_t	z1 = tree_cmz[tn1];

	nbcoord_t	res_x = 0.0;
	nbcoord_t	res_y = 0.0;
	nbcoord_t	res_z = 0.0;

	int stack_data[MAX_STACK_SIZE] = {};
	int	stack = 0;
	int	stack_head = stack;

	stack_data[stack++] = NBODY_HEAP_ROOT_INDEX;
	while(stack != stack_head)
	{
		int			curr = stack_data[--stack];
		nbcoord_t	dx = x1 - tree_cmx[curr];
		nbcoord_t	dy = y1 - tree_cmy[curr];
		nbcoord_t	dz = z1 - tree_cmz[curr];
		nbcoord_t	r2 = (dx * dx + dy * dy + dz * dz);

		if(r2 > tree_crit_r2[curr])
		{
			if(r2 < NBODY_MIN_R)
			{
				r2 = NBODY_MIN_R;
			}

			nbcoord_t	r = sqrt(r2);
			nbcoord_t	coeff = tree_mass[curr] / (r * r2);

			dx *= coeff;
			dy *= coeff;
			dz *= coeff;
			res_x -= dx;
			res_y -= dy;
			res_z -= dz;
		}
		else
		{
			int	left = left_idx(curr);
			int	rght = rght_idx(curr);
			if(left < tree_size)
			{
				stack_data[stack++] = left;
			}
			if(rght < tree_size)
			{
				stack_data[stack++] = rght;
			}
		}
	}

	__global const nbcoord_t*	vx = y + 3 * stride;
	__global const nbcoord_t*	vy = y + 4 * stride;
	__global const nbcoord_t*	vz = y + 5 * stride;

	__global nbcoord_t*	frx = f;
	__global nbcoord_t*	fry = frx + stride;
	__global nbcoord_t*	frz = frx + 2 * stride;
	__global nbcoord_t*	fvx = frx + 3 * stride;
	__global nbcoord_t*	fvy = frx + 4 * stride;
	__global nbcoord_t*	fvz = frx + 5 * stride;

	frx[n1] = vx[n1];
	fry[n1] = vy[n1];
	frz[n1] = vz[n1];
	fvx[n1] = res_x;
	fvy[n1] = res_y;
	fvz[n1] = res_z;
}

// Sparse fcompute using Kd-tree traverse (Barnes-Hut engine)
// Traverse starts form a tree node
// Stackless tree traverse
__kernel void ComputeHeapBHSL(int offset_n1, int points_count, int tree_size,
							  __global const nbcoord_t* y,
							  __global nbcoord_t* f,
							  __global const nbcoord_t* tree_cmx,
							  __global const nbcoord_t* tree_cmy,
							  __global const nbcoord_t* tree_cmz,
							  __global const nbcoord_t* tree_mass,
							  __global const nbcoord_t* tree_crit_r2,
							  __global const int* body_n)
{
	int		tree_offset = points_count - 1 + NBODY_HEAP_ROOT_INDEX;
	int		stride = points_count;
	int		tn1 = get_global_id(0) + offset_n1 + tree_offset;

	int			n1 = body_n[tn1];
	nbcoord_t	x1 = tree_cmx[tn1];
	nbcoord_t	y1 = tree_cmy[tn1];
	nbcoord_t	z1 = tree_cmz[tn1];

	nbcoord_t	res_x = 0.0;
	nbcoord_t	res_y = 0.0;
	nbcoord_t	res_z = 0.0;

	int	curr = NBODY_HEAP_ROOT_INDEX;
	do
	{
		nbcoord_t	dx = x1 - tree_cmx[curr];
		nbcoord_t	dy = y1 - tree_cmy[curr];
		nbcoord_t	dz = z1 - tree_cmz[curr];
		nbcoord_t	r2 = (dx * dx + dy * dy + dz * dz);

		if(r2 > tree_crit_r2[curr])
		{
			if(r2 < NBODY_MIN_R)
			{
				r2 = NBODY_MIN_R;
			}

			nbcoord_t	r = sqrt(r2);
			nbcoord_t	coeff = tree_mass[curr] / (r * r2);

			dx *= coeff;
			dy *= coeff;
			dz *= coeff;
			res_x -= dx;
			res_y -= dy;
			res_z -= dz;
			curr = next_down(curr);
		}
		else
		{
			curr = next_up(curr, tree_size);
		}
	}
	while(curr != NBODY_HEAP_ROOT_INDEX);

	__global const nbcoord_t*	vx = y + 3 * stride;
	__global const nbcoord_t*	vy = y + 4 * stride;
	__global const nbcoord_t*	vz = y + 5 * stride;

	__global nbcoord_t*	frx = f;
	__global nbcoord_t*	fry = frx + stride;
	__global nbcoord_t*	frz = frx + 2 * stride;
	__global nbcoord_t*	fvx = frx + 3 * stride;
	__global nbcoord_t*	fvy = frx + 4 * stride;
	__global nbcoord_t*	fvz = frx + 5 * stride;

	frx[n1] = vx[n1];
	fry[n1] = vy[n1];
	frz[n1] = vz[n1];
	fvx[n1] = res_x;
	fvy[n1] = res_y;
	fvz[n1] = res_z;
}

// Update leaf coordinates (Barnes-Hut engine)
__kernel void UpdateLeafBH(int points_count,
						   __global const nbcoord_t* y,
						   __global nbcoord_t* tree_cmx,
						   __global nbcoord_t* tree_cmy,
						   __global nbcoord_t* tree_cmz,
						   __global nbcoord_t* bmin_cmx,
						   __global nbcoord_t* bmin_cmy,
						   __global nbcoord_t* bmin_cmz,
						   __global nbcoord_t* bmax_cmx,
						   __global nbcoord_t* bmax_cmy,
						   __global nbcoord_t* bmax_cmz,
						   __global const int* body_n)
{
	int		tree_offset = points_count - 1 + NBODY_HEAP_ROOT_INDEX;
	int		stride = points_count;
	int		idx = get_global_id(0) + tree_offset;
	int		n = body_n[idx];
	bmin_cmx[idx] = bmax_cmx[idx] = tree_cmx[idx] = y[0 * stride + n];
	bmin_cmy[idx] = bmax_cmy[idx] = tree_cmy[idx] = y[1 * stride + n];
	bmin_cmz[idx] = bmax_cmz[idx] = tree_cmz[idx] = y[2 * stride + n];
}

// Update node coordinates (Barnes-Hut engine)
__kernel void UpdateNodeBH(int level_size,
						   __global nbcoord_t* tree_cmx,
						   __global nbcoord_t* tree_cmy,
						   __global nbcoord_t* tree_cmz,
						   __global nbcoord_t* bmin_cmx,
						   __global nbcoord_t* bmin_cmy,
						   __global nbcoord_t* bmin_cmz,
						   __global nbcoord_t* bmax_cmx,
						   __global nbcoord_t* bmax_cmy,
						   __global nbcoord_t* bmax_cmz,
						   __global nbcoord_t* tree_mass,
						   __global nbcoord_t* tree_crit_r2,
						   nbcoord_t distance_to_node_radius_ratio_sqr)
{
	int		idx = get_global_id(0) + level_size;
	if(idx >= 2 * level_size)
	{
		return;
	}
	int		left = left_idx(idx);
	int		rght = rght_idx(idx);
	nbcoord_t	mass = tree_mass[left] + tree_mass[rght];
	nbcoord3_t	mass_center = {(tree_cmx[left] * tree_mass[left] +
								tree_cmx[rght] * tree_mass[rght]) / mass,
							   (tree_cmy[left] * tree_mass[left] +
								tree_cmy[rght] * tree_mass[rght]) / mass,
							   (tree_cmz[left] * tree_mass[left] +
								tree_cmz[rght] * tree_mass[rght]) / mass
							 };
	tree_mass[idx] = mass;

	tree_cmx[idx] = mass_center.x;
	tree_cmy[idx] = mass_center.y;
	tree_cmz[idx] = mass_center.z;
	nbcoord3_t	bmin = {min(bmin_cmx[left], bmin_cmx[rght]),
						min(bmin_cmy[left], bmin_cmy[rght]),
						min(bmin_cmz[left], bmin_cmz[rght])
					  };
	nbcoord3_t	bmax = {max(bmax_cmx[left], bmax_cmx[rght]),
						max(bmax_cmy[left], bmax_cmy[rght]),
						max(bmax_cmz[left], bmax_cmz[rght])
					  };
	bmin_cmx[idx] = bmin.x;
	bmin_cmy[idx] = bmin.y;
	bmin_cmz[idx] = bmin.z;
	bmax_cmx[idx] = bmax.x;
	bmax_cmy[idx] = bmax.y;
	bmax_cmz[idx] = bmax.z;

	nbcoord_t	r = distance(bmin, bmax) / 2 +
					distance((bmin + bmax) / 2, mass_center);
	tree_crit_r2[idx] = (r * r) * distance_to_node_radius_ratio_sqr;
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
__kernel void fmadd_inplace(int offset, __global nbcoord_t* a,
							__global const nbcoord_t* b, nbcoord_t c)
{
	int		i = get_global_id(0) + offset;
	a[i] += b[i] * c;
}

//! a[i] += b[i]*c with correction
__kernel void fmadd_inplace_corr(int offset, __global nbcoord_t* _a, __global nbcoord_t* corr,
								 __global const nbcoord_t* b, nbcoord_t c)
{
	int			i = get_global_id(0) + offset;
	nbcoord_t	term = b[i] * c;
	nbcoord_t	a = _a[i];
	nbcoord_t	corrected = term - corr[i];
	nbcoord_t	new_sum = a + corrected;

	corr[i] = (new_sum - a) - corrected;
	_a[i] =  new_sum;
}

//! a[i+aoff] = b[i+boff] + c[i+coff]*d
__kernel void fmadd(__global nbcoord_t* a, __global const nbcoord_t* b, __global const nbcoord_t* c,
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

