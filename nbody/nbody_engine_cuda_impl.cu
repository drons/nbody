#include "nbody_engine_cuda_impl.h"

#define NB_CALL_TYPE __device__ inline

#include "nbody_space_heap_func.h"

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

__global__ void kfcompute_xyz(const nbcoord_t* y, int yoff, nbcoord_t* f, int foff, int stride)
{
	int n1 = blockDim.x * blockIdx.x + threadIdx.x;

	const nbcoord_t*	vx = y + yoff + 3 * stride;
	const nbcoord_t*	vy = vx + stride;
	const nbcoord_t*	vz = vx + 2 * stride;

	nbcoord_t*	frx = f + foff;
	nbcoord_t*	fry = frx + stride;
	nbcoord_t*	frz = frx + 2 * stride;

	frx[n1] = vx[n1];
	fry[n1] = vy[n1];
	frz[n1] = vz[n1];
}

__host__ void fcompute_xyz(const nbcoord_t* y, nbcoord_t* f, int count, int block_size)
{
	dim3 grid(count / block_size);
	dim3 block(block_size);

	kfcompute_xyz <<< grid, block >>> (y, 0, f, 0, count);
}

__global__ void kfcompute(int offset_n2, const nbcoord_t* y, int yoff, nbcoord_t* f, int foff,
						  const nbcoord_t* mass, int points_count, int stride)
{
	int n1 = blockDim.x * blockIdx.x + threadIdx.x;

	const nbcoord_t*	rx = y + yoff;
	const nbcoord_t*	ry = rx + stride;
	const nbcoord_t*	rz = rx + 2 * stride;

	nbcoord_t	x1 = rx[n1];
	nbcoord_t	y1 = ry[n1];
	nbcoord_t	z1 = rz[n1];

	nbcoord_t	res_x = 0.0;
	nbcoord_t	res_y = 0.0;
	nbcoord_t	res_z = 0.0;

	__shared__ nbcoord_t	x2[NBODY_DATA_BLOCK_SIZE];
	__shared__ nbcoord_t	y2[NBODY_DATA_BLOCK_SIZE];
	__shared__ nbcoord_t	z2[NBODY_DATA_BLOCK_SIZE];
	__shared__ nbcoord_t	m2[NBODY_DATA_BLOCK_SIZE];

	// NB! get_local_size(0) == NBODY_DATA_BLOCK_SIZE
	for(int b2 = 0; b2 < points_count; b2 += NBODY_DATA_BLOCK_SIZE)
	{
		int			n2 = b2 + offset_n2 + threadIdx.x;

		// Copy data block to local memory
		x2[ threadIdx.x ] = rx[n2];
		y2[ threadIdx.x ] = ry[n2];
		z2[ threadIdx.x ] = rz[n2];
		m2[ threadIdx.x ] = mass[n2];

		// Synchronize local work-items copy operations
		__syncthreads();

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
		__syncthreads();

		res_x += local_res_x;
		res_y += local_res_y;
		res_z += local_res_z;
	}

	n1 += foff;
	f[n1 + 3 * stride] = res_x;
	f[n1 + 4 * stride] = res_y;
	f[n1 + 5 * stride] = res_z;
}

__host__ void fcompute_block(const nbcoord_t* y, nbcoord_t* f, const nbcoord_t* m,
							 int count, int block_size)
{
	dim3 grid(count / block_size);
	dim3 block(block_size);

	kfcompute <<< grid, block >>> (0, y, 0, f, 0, m, count, count);
}

#define MAX_STACK_SIZE 24

// Sparse fcompute using Kd-tree traverse (Barnes-Hut engine)
// Traverse starts form a tree node
__global__ void kfcompute_heap_bh(int offset_n1, int points_count, int tree_size,
								  nbcoord_t* f,
								  const nbcoord_t* tree_cmx,
								  const nbcoord_t* tree_cmy,
								  const nbcoord_t* tree_cmz,
								  const nbcoord_t* tree_mass,
								  const nbcoord_t* tree_crit_r2,
								  const int* body_n)
{
	int		tree_offset = points_count - 1;
	int		stride = points_count;
	int		tn1 = blockDim.x * blockIdx.x + threadIdx.x + offset_n1 + tree_offset;

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

	stack_data[stack++] = 0;
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
			int	left = nbody_heap_func<int>::left_idx(curr);
			int	rght = nbody_heap_func<int>::rght_idx(curr);
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

	f[n1 + 3 * stride] = res_x;
	f[n1 + 4 * stride] = res_y;
	f[n1 + 5 * stride] = res_z;
}

__host__ void fcompute_heap_bh(int offset_n1, int points_count, int tree_size,
							   nbcoord_t* f,
							   const nbcoord_t* tree_cmx,
							   const nbcoord_t* tree_cmy,
							   const nbcoord_t* tree_cmz,
							   const nbcoord_t* tree_mass,
							   const nbcoord_t* tree_crit_r2,
							   const int* body_n,
							   int block_size)
{
	dim3 grid(points_count / block_size);
	dim3 block(block_size);

	cudaFuncSetCacheConfig(kfcompute_heap_bh, cudaFuncCachePreferL1);

	kfcompute_heap_bh <<< grid, block >>> (offset_n1, points_count, tree_size, f,
										   tree_cmx, tree_cmy, tree_cmz, tree_mass,
										   tree_crit_r2, body_n);
}

// Sparse fcompute using Kd-tree traverse (Barnes-Hut engine)
// Traverse starts form a tree node
// Tree is stored in texture memory (float x = tex1Dfetch<float>(tex, i);)

template<typename T>
struct nb1Dfetch
{
};
template<>
struct nb1Dfetch<double>
{
	__device__ double fetch(cudaTextureObject_t tex, int i)
	{
		int2 p(tex1Dfetch<int2>(tex, i));
		return __hiloint2double(p.y, p.x);
	}
};
template<>
struct nb1Dfetch<float>
{
	__device__ float fetch(cudaTextureObject_t tex, int i)
	{
		return tex1Dfetch<float>(tex, i);
	}
};

__global__ void kfcompute_heap_bh_tex(int offset_n1, int points_count, int tree_size,
									  nbcoord_t* f,
									  cudaTextureObject_t tree_cmx,
									  cudaTextureObject_t tree_cmy,
									  cudaTextureObject_t tree_cmz,
									  cudaTextureObject_t tree_mass,
									  cudaTextureObject_t tree_crit_r2,
									  const int* body_n)
{
	nb1Dfetch<nbcoord_t>	tex;
	int		tree_offset = points_count - 1;
	int		stride = points_count;
	int		tn1 = blockDim.x * blockIdx.x + threadIdx.x + offset_n1 + tree_offset;

	int			n1 = body_n[tn1];
	nbcoord_t	x1 = tex.fetch(tree_cmx, tn1);
	nbcoord_t	y1 = tex.fetch(tree_cmy, tn1);
	nbcoord_t	z1 = tex.fetch(tree_cmz, tn1);

	nbcoord_t	res_x = 0.0;
	nbcoord_t	res_y = 0.0;
	nbcoord_t	res_z = 0.0;

	int stack_data[MAX_STACK_SIZE] = {};
	int	stack = 0;
	int	stack_head = stack;

	stack_data[stack++] = 0;
	while(stack != stack_head)
	{
		int			curr = stack_data[--stack];
		nbcoord_t	dx = x1 - tex.fetch(tree_cmx, curr);
		nbcoord_t	dy = y1 - tex.fetch(tree_cmy, curr);
		nbcoord_t	dz = z1 - tex.fetch(tree_cmz, curr);
		nbcoord_t	r2 = (dx * dx + dy * dy + dz * dz);

		if(r2 > tex.fetch(tree_crit_r2, curr))
		{
			if(r2 < NBODY_MIN_R)
			{
				r2 = NBODY_MIN_R;
			}

			nbcoord_t	r = sqrt(r2);
			nbcoord_t	coeff = tex.fetch(tree_mass, curr) / (r * r2);

			dx *= coeff;
			dy *= coeff;
			dz *= coeff;
			res_x -= dx;
			res_y -= dy;
			res_z -= dz;
		}
		else
		{
			int	left = nbody_heap_func<int>::left_idx(curr);
			int	rght = nbody_heap_func<int>::rght_idx(curr);
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

	f[n1 + 3 * stride] = res_x;
	f[n1 + 4 * stride] = res_y;
	f[n1 + 5 * stride] = res_z;
}

__host__ void fcompute_heap_bh_tex(int offset_n1, int points_count, int tree_size,
								   nbcoord_t* f,
								   cudaTextureObject_t tree_cmx,
								   cudaTextureObject_t tree_cmy,
								   cudaTextureObject_t tree_cmz,
								   cudaTextureObject_t tree_mass,
								   cudaTextureObject_t tree_crit_r2,
								   const int* body_n,
								   int block_size)
{
	dim3 grid(points_count / block_size);
	dim3 block(block_size);

	cudaFuncSetCacheConfig(kfcompute_heap_bh_tex, cudaFuncCachePreferL1);

	kfcompute_heap_bh_tex <<< grid, block >>> (offset_n1, points_count, tree_size, f,
											   tree_cmx, tree_cmy, tree_cmz, tree_mass,
											   tree_crit_r2, body_n);
}

__global__ void kfcompute_heap_bh_stackless(int offset_n1, int points_count, int tree_size,
											nbcoord_t* f,
											cudaTextureObject_t tree_cmx,
											cudaTextureObject_t tree_cmy,
											cudaTextureObject_t tree_cmz,
											cudaTextureObject_t tree_mass,
											cudaTextureObject_t tree_crit_r2,
											const int* body_n)
{
	nb1Dfetch<nbcoord_t>	tex;
	int		tree_offset = points_count - 1;
	int		stride = points_count;
	int		tn1 = blockDim.x * blockIdx.x + threadIdx.x + offset_n1 + tree_offset;

	int			n1 = body_n[tn1];
	nbcoord_t	x1 = tex.fetch(tree_cmx, tn1);
	nbcoord_t	y1 = tex.fetch(tree_cmy, tn1);
	nbcoord_t	z1 = tex.fetch(tree_cmz, tn1);

	nbcoord_t	res_x = 0.0;
	nbcoord_t	res_y = 0.0;
	nbcoord_t	res_z = 0.0;

	int	curr = 0;
	do
	{
		nbcoord_t	dx = x1 - tex.fetch(tree_cmx, curr);
		nbcoord_t	dy = y1 - tex.fetch(tree_cmy, curr);
		nbcoord_t	dz = z1 - tex.fetch(tree_cmz, curr);
		nbcoord_t	r2 = (dx * dx + dy * dy + dz * dz);

		if(r2 > tex.fetch(tree_crit_r2, curr))
		{
			if(r2 < NBODY_MIN_R)
			{
				r2 = NBODY_MIN_R;
			}

			nbcoord_t	r = sqrt(r2);
			nbcoord_t	coeff = tex.fetch(tree_mass, curr) / (r * r2);

			dx *= coeff;
			dy *= coeff;
			dz *= coeff;
			res_x -= dx;
			res_y -= dy;
			res_z -= dz;
			curr = nbody_heap_func<int>::skip_idx(curr);
		}
		else
		{
			curr = nbody_heap_func<int>::next_up(curr, tree_size);
		}
	}
	while(curr != 0);

	f[n1 + 3 * stride] = res_x;
	f[n1 + 4 * stride] = res_y;
	f[n1 + 5 * stride] = res_z;
}

__host__ void fcompute_heap_bh_stackless(int offset_n1, int points_count, int tree_size,
										 nbcoord_t* f,
										 cudaTextureObject_t tree_cmx,
										 cudaTextureObject_t tree_cmy,
										 cudaTextureObject_t tree_cmz,
										 cudaTextureObject_t tree_mass,
										 cudaTextureObject_t tree_crit_r2,
										 const int* body_n,
										 int block_size)
{
	dim3 grid(points_count / block_size);
	dim3 block(block_size);

	cudaFuncSetCacheConfig(kfcompute_heap_bh_stackless, cudaFuncCachePreferL1);

	kfcompute_heap_bh_stackless <<< grid, block >>> (offset_n1, points_count, tree_size, f,
													 tree_cmx, tree_cmy, tree_cmz, tree_mass,
													 tree_crit_r2, body_n);
}

__host__ void fill_buffer(nbcoord_t* dev_ptr, nbcoord_t v, int count)
{
	thrust::device_ptr<nbcoord_t>	ptr(thrust::device_pointer_cast(dev_ptr));

	thrust::fill(ptr, ptr + count, v);
}

//! a[i] += b[i]*c
__global__ void kfmadd_inplace(int offset, nbcoord_t* a, const nbcoord_t* b, nbcoord_t c)
{
	int		i = blockDim.x * blockIdx.x + threadIdx.x + offset;
	a[i] += b[i] * c;
}

__host__ void fmadd_inplace(int offset, nbcoord_t* a, const nbcoord_t* b, nbcoord_t c, int count)
{
	dim3 grid(count / NBODY_DATA_BLOCK_SIZE);
	dim3 block(NBODY_DATA_BLOCK_SIZE);

	kfmadd_inplace <<< grid, block >>> (offset, a, b, c);
}

//! a[i+aoff] = b[i+boff] + c[i+coff]*d
__global__ void kfmadd(nbcoord_t* a, const nbcoord_t* b, const nbcoord_t* c,
					   nbcoord_t d, int aoff, int boff, int coff)
{
	int		i = blockDim.x * blockIdx.x + threadIdx.x;
	a[i + aoff] = b[i + boff] + c[i + coff] * d;
}

__host__ void fmadd(nbcoord_t* a, const nbcoord_t* b, const nbcoord_t* c,
					nbcoord_t d, int aoff, int boff, int coff, int count)
{
	dim3 grid(count / NBODY_DATA_BLOCK_SIZE);
	dim3 block(NBODY_DATA_BLOCK_SIZE);

	kfmadd <<< grid, block >>> (a, b, c, d, aoff, boff, coff);
}

//! @result = max( fabs(a[k]), k=[0...asize) )
__host__ void fmaxabs(const nbcoord_t* a, int asize, nbcoord_t& result)
{
	thrust::device_ptr<const nbcoord_t>	ptr(thrust::device_pointer_cast(a));

	thrust::pair<thrust::device_ptr<const nbcoord_t>, thrust::device_ptr<const nbcoord_t> > minmax =
		thrust::minmax_element(ptr, ptr + asize);

	result = std::max(std::abs(minmax.first[0]),
					  std::abs(minmax.second[0]));
}

