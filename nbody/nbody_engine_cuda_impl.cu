#include "nbody_engine_cuda_impl.h"

#define NB_CALL_TYPE static __device__ inline

#include "nbody_space_heap_func.h"

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

__global__ void kfcompute_xyz(const nbcoord_t* y, nbcoord_t* f, int stride)
{
	int n1 = blockDim.x * blockIdx.x + threadIdx.x;

	const nbcoord_t*	vx = y + 3 * stride;
	const nbcoord_t*	vy = vx + stride;
	const nbcoord_t*	vz = vx + 2 * stride;

	nbcoord_t*	frx = f;
	nbcoord_t*	fry = frx + stride;
	nbcoord_t*	frz = frx + 2 * stride;

	frx[n1] = vx[n1];
	fry[n1] = vy[n1];
	frz[n1] = vz[n1];
}

__host__ void fcompute_xyz(const nbcoord_t* y, nbcoord_t* f, size_t count,
						   size_t stride, int block_size)
{
	dim3 grid(count / block_size);
	dim3 block(block_size);

	kfcompute_xyz <<< grid, block >>> (y, f, stride);
}

__global__ void kfcompute(int offset_n1, const nbcoord_t* y, nbcoord_t* f,
						  const nbcoord_t* mass, int points_count, int stride)
{
	int n1 = blockDim.x * blockIdx.x + threadIdx.x + offset_n1;

	const nbcoord_t*	rx = y;
	const nbcoord_t*	ry = rx + stride;
	const nbcoord_t*	rz = rx + 2 * stride;

	nbcoord_t	x1 = rx[n1];
	nbcoord_t	y1 = ry[n1];
	nbcoord_t	z1 = rz[n1];

	nbcoord_t	res_x = 0.0;
	nbcoord_t	res_y = 0.0;
	nbcoord_t	res_z = 0.0;

	extern __shared__ nbcoord_t shared_xyzm_buf[];

	nbcoord_t*	x2 = shared_xyzm_buf;
	nbcoord_t*	y2 = x2 + blockDim.x;
	nbcoord_t*	z2 = y2 + blockDim.x;
	nbcoord_t*	m2 = z2 + blockDim.x;

	for(int b2 = 0; b2 < points_count; b2 += blockDim.x)
	{
		int			n2 = b2 + threadIdx.x;

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

		for(int n2 = 0; n2 != blockDim.x; ++n2)
		{
			nbcoord_t	dx = x1 - x2[n2];
			nbcoord_t	dy = y1 - y2[n2];
			nbcoord_t	dz = z1 - z2[n2];
			nbcoord_t	r2 = (dx * dx + dy * dy + dz * dz);

			if(r2 < nbody::MinDistance)
			{
				r2 = nbody::MinDistance;
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

	f[n1 + 3 * stride] = res_x;
	f[n1 + 4 * stride] = res_y;
	f[n1 + 5 * stride] = res_z;
}

__host__ void fcompute_block(size_t off, const nbcoord_t* y, nbcoord_t* f, const nbcoord_t* m,
							 size_t count, size_t total_count, int block_size)
{
	dim3	grid(count / block_size);
	dim3	block(block_size);
	size_t	shared_size(4 * sizeof(nbcoord_t) * block_size);

	kfcompute <<< grid, block, shared_size >>> (off, y, f, m, total_count, total_count);
}

#define MAX_STACK_SIZE 64

// Sparse fcompute using Kd-tree traverse (Barnes-Hut engine)
// Traverse starts form a tree node
__global__ void kfcompute_heap_bh(int offset_n1, int points_count, int tree_size,
								  const nbcoord_t* y,
								  nbcoord_t* f,
								  const nbcoord_t* tree_cmx,
								  const nbcoord_t* tree_cmy,
								  const nbcoord_t* tree_cmz,
								  const nbcoord_t* tree_mass,
								  const nbcoord_t* tree_crit_r2,
								  const int* body_n)
{
	int		tree_offset = points_count - 1 + NBODY_HEAP_ROOT_INDEX;
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
			if(r2 < nbody::MinDistance)
			{
				r2 = nbody::MinDistance;
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

	f[n1 + 0 * stride] = y[n1 + 3 * stride];
	f[n1 + 1 * stride] = y[n1 + 4 * stride];
	f[n1 + 2 * stride] = y[n1 + 5 * stride];
	f[n1 + 3 * stride] = res_x;
	f[n1 + 4 * stride] = res_y;
	f[n1 + 5 * stride] = res_z;
}

__host__ void fcompute_heap_bh(int offset_n1, int points_count,
							   int compute_points_count, int tree_size,
							   const nbcoord_t* y,
							   nbcoord_t* f,
							   const nbcoord_t* tree_cmx,
							   const nbcoord_t* tree_cmy,
							   const nbcoord_t* tree_cmz,
							   const nbcoord_t* tree_mass,
							   const nbcoord_t* tree_crit_r2,
							   const int* body_n,
							   int block_size)
{
	dim3 grid(compute_points_count / block_size);
	dim3 block(block_size);

	cudaFuncSetCacheConfig(kfcompute_heap_bh, cudaFuncCachePreferL1);

	kfcompute_heap_bh <<< grid, block >>> (offset_n1, points_count, tree_size, y, f,
										   tree_cmx, tree_cmy, tree_cmz,
										   tree_mass, tree_crit_r2, body_n);
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
	typedef double3 vec3;
	typedef double4 vec4;
	static __device__ double fetch(cudaTextureObject_t tex, int i)
	{
		int2 p(tex1Dfetch<int2>(tex, i));
		return __hiloint2double(p.y, p.x);
	}
	static __device__ vec4 fetch4(cudaTextureObject_t tex, int i)
	{
		int		ii(2 * i);
		int4	p1(tex1Dfetch<int4>(tex, ii));
		int4	p2(tex1Dfetch<int4>(tex, ii + 1));
		vec4	d4 = {__hiloint2double(p1.y, p1.x),
					  __hiloint2double(p1.w, p1.z),
					  __hiloint2double(p2.y, p2.x),
					  __hiloint2double(p2.w, p2.z)
				  };
		return d4;
	}
};
template<>
struct nb1Dfetch<float>
{
	typedef float3 vec3;
	typedef float4 vec4;
	static __device__ float fetch(cudaTextureObject_t tex, int i)
	{
		return tex1Dfetch<float>(tex, i);
	}
	static __device__ vec4 fetch4(cudaTextureObject_t tex, int i)
	{
		return tex1Dfetch<float4>(tex, i);
	}
};

typedef nb1Dfetch<nbcoord_t>::vec3 nbvec3_t;
typedef nb1Dfetch<nbcoord_t>::vec4 nbvec4_t;

__global__ void kfcompute_heap_bh_tex(int offset_n1, int points_count, int tree_size,
									  const nbcoord_t* y, nbcoord_t* f,
									  cudaTextureObject_t tree_xyzr,
									  cudaTextureObject_t tree_mass,
									  const int* body_n)
{
	nb1Dfetch<nbcoord_t>	tex;
	int		tree_offset = points_count - 1 + NBODY_HEAP_ROOT_INDEX;
	int		stride = points_count;
	int		tn1 = blockDim.x * blockIdx.x + threadIdx.x + offset_n1 + tree_offset;

	int			n1 = body_n[tn1];
	nbvec4_t	xyzr = tex.fetch4(tree_xyzr, tn1);
	nbcoord_t	x1 = xyzr.x;
	nbcoord_t	y1 = xyzr.y;
	nbcoord_t	z1 = xyzr.z;

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
		nbvec4_t	xyzr2 = tex.fetch4(tree_xyzr, curr);
		nbcoord_t	dx = x1 - xyzr2.x;
		nbcoord_t	dy = y1 - xyzr2.y;
		nbcoord_t	dz = z1 - xyzr2.z;
		nbcoord_t	r2 = (dx * dx + dy * dy + dz * dz);

		if(r2 > xyzr2.w)
		{
			if(r2 < nbody::MinDistance)
			{
				r2 = nbody::MinDistance;
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

	f[n1 + 0 * stride] = y[n1 + 3 * stride];
	f[n1 + 1 * stride] = y[n1 + 4 * stride];
	f[n1 + 2 * stride] = y[n1 + 5 * stride];
	f[n1 + 3 * stride] = res_x;
	f[n1 + 4 * stride] = res_y;
	f[n1 + 5 * stride] = res_z;
}

__host__ void fcompute_heap_bh_tex(int offset_n1, int points_count,
								   int compute_points_count, int tree_size,
								   const nbcoord_t* y,
								   nbcoord_t* f,
								   cudaTextureObject_t tree_xyzr,
								   cudaTextureObject_t tree_mass,
								   const int* body_n,
								   int block_size)
{
	dim3 grid(compute_points_count / block_size);
	dim3 block(block_size);

	cudaFuncSetCacheConfig(kfcompute_heap_bh_tex, cudaFuncCachePreferL1);

	kfcompute_heap_bh_tex <<< grid, block >>> (offset_n1, points_count, tree_size, y, f,
											   tree_xyzr, tree_mass, body_n);
}

__global__ void kfcompute_heap_bh_stackless(int offset_n1, int points_count, int tree_size,
											const nbcoord_t* y, nbcoord_t* f,
											cudaTextureObject_t tree_xyzr,
											cudaTextureObject_t tree_mass,
											const int* body_n)
{
	nb1Dfetch<nbcoord_t>	tex;
	int		tree_offset = points_count - 1 + NBODY_HEAP_ROOT_INDEX;
	int		stride = points_count;
	int		tn1 = blockDim.x * blockIdx.x + threadIdx.x + offset_n1 + tree_offset;

	int			n1 = body_n[tn1];
	nbvec4_t	xyzr = tex.fetch4(tree_xyzr, tn1);
	nbcoord_t	x1 = xyzr.x;
	nbcoord_t	y1 = xyzr.y;
	nbcoord_t	z1 = xyzr.z;

	nbcoord_t	res_x = 0.0;
	nbcoord_t	res_y = 0.0;
	nbcoord_t	res_z = 0.0;

	int	curr = NBODY_HEAP_ROOT_INDEX;
	do
	{
		nbvec4_t	xyzr2 = tex.fetch4(tree_xyzr, curr);
		nbcoord_t	dx = x1 - xyzr2.x;
		nbcoord_t	dy = y1 - xyzr2.y;
		nbcoord_t	dz = z1 - xyzr2.z;
		nbcoord_t	r2 = (dx * dx + dy * dy + dz * dz);

		if(r2 > xyzr2.w)
		{
			if(r2 < nbody::MinDistance)
			{
				r2 = nbody::MinDistance;
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
	while(curr != NBODY_HEAP_ROOT_INDEX);//NOLINT

	f[n1 + 0 * stride] = y[n1 + 3 * stride];
	f[n1 + 1 * stride] = y[n1 + 4 * stride];
	f[n1 + 2 * stride] = y[n1 + 5 * stride];
	f[n1 + 3 * stride] = res_x;
	f[n1 + 4 * stride] = res_y;
	f[n1 + 5 * stride] = res_z;
}

__host__ void fcompute_heap_bh_stackless(int offset_n1, int points_count,
										 int compute_points_count, int tree_size,
										 const nbcoord_t* y, nbcoord_t* f,
										 cudaTextureObject_t tree_xyzr,
										 cudaTextureObject_t tree_mass,
										 const int* body_n,
										 int block_size)
{
	dim3 grid(compute_points_count / block_size);
	dim3 block(block_size);

	cudaFuncSetCacheConfig(kfcompute_heap_bh_stackless, cudaFuncCachePreferL1);

	kfcompute_heap_bh_stackless <<< grid, block >>> (offset_n1, points_count, tree_size, y, f,
													 tree_xyzr, tree_mass, body_n);
	//printf("%s\n", cudaGetErrorString(cudaGetLastError()));
}

inline __host__ __device__ nbcoord_t distance(nbvec3_t a, nbvec3_t b)
{
	nbcoord_t	dx = a.x - b.x;
	nbcoord_t	dy = a.y - b.y;
	nbcoord_t	dz = a.z - b.z;
	return sqrt(dx * dx + dy * dy + dz * dz);
}

inline __device__ nbvec3_t operator +(nbvec3_t a, nbvec3_t b)
{
	return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline __device__ nbvec3_t operator -(nbvec3_t a, nbvec3_t b)
{
	return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline __device__ nbvec3_t operator /(nbvec3_t a, nbcoord_t b)
{
	return {a.x / b, a.y / b, a.z / b};
}

// Update leaf coordinates (Barnes-Hut engine)
__global__ void kupdate_leaf_bh(int points_count,
								const nbcoord_t* y,
								nbcoord_t* tree_cmx,
								nbcoord_t* tree_cmy,
								nbcoord_t* tree_cmz,
								nbcoord_t* bmin_cmx,
								nbcoord_t* bmin_cmy,
								nbcoord_t* bmin_cmz,
								nbcoord_t* bmax_cmx,
								nbcoord_t* bmax_cmy,
								nbcoord_t* bmax_cmz,
								const int* body_n)
{
	int		tree_offset = points_count - 1 + NBODY_HEAP_ROOT_INDEX;
	int		stride = points_count;
	int		idx = blockDim.x * blockIdx.x + threadIdx.x + tree_offset;
	int		n = body_n[idx];
	bmin_cmx[idx] = bmax_cmx[idx] = tree_cmx[idx] = y[0 * stride + n];
	bmin_cmy[idx] = bmax_cmy[idx] = tree_cmy[idx] = y[1 * stride + n];
	bmin_cmz[idx] = bmax_cmz[idx] = tree_cmz[idx] = y[2 * stride + n];
}

__host__ void update_leaf_bh(int points_count,
							 const nbcoord_t* y,
							 nbcoord_t* tree_cmx,
							 nbcoord_t* tree_cmy,
							 nbcoord_t* tree_cmz,
							 nbcoord_t* bmin_cmx,
							 nbcoord_t* bmin_cmy,
							 nbcoord_t* bmin_cmz,
							 nbcoord_t* bmax_cmx,
							 nbcoord_t* bmax_cmy,
							 nbcoord_t* bmax_cmz,
							 const int* body_n)
{
	dim3 grid(points_count / NBODY_DATA_BLOCK_SIZE);
	dim3 block(NBODY_DATA_BLOCK_SIZE);

	kupdate_leaf_bh <<< grid, block >>> (points_count, y, tree_cmx, tree_cmy, tree_cmz,
										 bmin_cmx, bmin_cmy, bmin_cmz,
										 bmax_cmx, bmax_cmy, bmax_cmz, body_n);
}

// Update node coordinates (Barnes-Hut engine)
__global__ void kupdate_node_bh(int level_size,
								nbcoord_t* tree_cmx,
								nbcoord_t* tree_cmy,
								nbcoord_t* tree_cmz,
								nbcoord_t* bmin_cmx,
								nbcoord_t* bmin_cmy,
								nbcoord_t* bmin_cmz,
								nbcoord_t* bmax_cmx,
								nbcoord_t* bmax_cmy,
								nbcoord_t* bmax_cmz,
								nbcoord_t* tree_mass,
								nbcoord_t* tree_crit_r2,
								nbcoord_t distance_to_node_radius_ratio_sqr)
{
	int		idx = blockDim.x * blockIdx.x + threadIdx.x + level_size;
	if(idx >= 2 * level_size)
	{
		return;
	}
	int		left = nbody_heap_func<int>::left_idx(idx);
	int		rght = nbody_heap_func<int>::rght_idx(idx);
	nbcoord_t	mass = tree_mass[left] + tree_mass[rght];
	nbvec3_t	mass_center = {(tree_cmx[left] * tree_mass[left] +
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
	nbvec3_t	bmin = {min(bmin_cmx[left], bmin_cmx[rght]),
						min(bmin_cmy[left], bmin_cmy[rght]),
						min(bmin_cmz[left], bmin_cmz[rght])
					};
	nbvec3_t	bmax = {max(bmax_cmx[left], bmax_cmx[rght]),
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

__host__ void update_node_bh(int level_size,
							 nbcoord_t* tree_cmx,
							 nbcoord_t* tree_cmy,
							 nbcoord_t* tree_cmz,
							 nbcoord_t* bmin_cmx,
							 nbcoord_t* bmin_cmy,
							 nbcoord_t* bmin_cmz,
							 nbcoord_t* bmax_cmx,
							 nbcoord_t* bmax_cmy,
							 nbcoord_t* bmax_cmz,
							 nbcoord_t* tree_mass,
							 nbcoord_t* tree_crit_r2,
							 nbcoord_t distance_to_node_radius_ratio_sqr)
{
	dim3 grid(std::max(1, level_size / NBODY_DATA_BLOCK_SIZE));
	dim3 block(NBODY_DATA_BLOCK_SIZE);

	kupdate_node_bh <<< grid, block >>> (level_size,
										 tree_cmx, tree_cmy, tree_cmz,
										 bmin_cmx, bmin_cmy, bmin_cmz,
										 bmax_cmx, bmax_cmy, bmax_cmz,
										 tree_mass, tree_crit_r2,
										 distance_to_node_radius_ratio_sqr);
}

__host__ void fill_buffer(nbcoord_t* dev_ptr, nbcoord_t v, int count)
{
	thrust::device_ptr<nbcoord_t>	ptr(thrust::device_pointer_cast(dev_ptr));

	thrust::fill(ptr, ptr + count, v);
}

//! a[i] += b[i]*c
__global__ void kfmadd_inplace(nbcoord_t* a, const nbcoord_t* b, nbcoord_t c)
{
	int		i = blockDim.x * blockIdx.x + threadIdx.x;
	a[i] += b[i] * c;
}

__host__ void fmadd_inplace(nbcoord_t* a, const nbcoord_t* b, nbcoord_t c, int count)
{
	dim3 grid(count / NBODY_DATA_BLOCK_SIZE);
	dim3 block(NBODY_DATA_BLOCK_SIZE);

	kfmadd_inplace <<< grid, block >>> (a, b, c);
}

//! a[i] += b[i]*c with correction
__global__ void kfmadd_inplace_corr(nbcoord_t* _a, nbcoord_t* corr,
									const nbcoord_t* b, nbcoord_t c)
{
	int		i = blockDim.x * blockIdx.x + threadIdx.x;
	nbcoord_t	term = b[i] * c;
	nbcoord_t	a = _a[i];
	nbcoord_t	corrected = term - corr[i];
	nbcoord_t	new_sum = a + corrected;

	corr[i] = (new_sum - a) - corrected;
	_a[i] =  new_sum;
}

__host__ void fmadd_inplace_corr(nbcoord_t* a, nbcoord_t* corr,
								 const nbcoord_t* b, nbcoord_t c, int count)
{
	dim3 grid(count / NBODY_DATA_BLOCK_SIZE);
	dim3 block(NBODY_DATA_BLOCK_SIZE);

	kfmadd_inplace_corr <<< grid, block >>> (a, corr, b, c);
}

//! a[i+aoff] = b[i+boff] + c[i+coff]*d
__global__ void kfmadd(nbcoord_t* a, const nbcoord_t* b, const nbcoord_t* c,
					   nbcoord_t d)
{
	int		i = blockDim.x * blockIdx.x + threadIdx.x;
	a[i] = b[i] + c[i] * d;
}

__host__ void fmadd(nbcoord_t* a, const nbcoord_t* b, const nbcoord_t* c,
					nbcoord_t d, int count)
{
	dim3 grid(count / NBODY_DATA_BLOCK_SIZE);
	dim3 block(NBODY_DATA_BLOCK_SIZE);

	kfmadd <<< grid, block >>> (a, b, c, d);
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

