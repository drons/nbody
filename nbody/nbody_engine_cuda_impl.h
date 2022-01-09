#ifndef NBODY_ENGINE_CUDA_IMPL_H
#define NBODY_ENGINE_CUDA_IMPL_H

#include <cuda_runtime.h>
#include "nbtype.h"

__host__ void fcompute_xyz(const nbcoord_t* y, nbcoord_t* f, size_t count,
						   size_t stride, int block_size);
__host__ void fcompute_block(size_t off, const nbcoord_t* y, nbcoord_t* f, const nbcoord_t* m,
							 size_t count, size_t total_count, int block_size);
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
							   int block_size);

__host__ void fcompute_heap_bh_tex(int offset_n1, int points_count, int tree_size,
								   nbcoord_t* f,
								   cudaTextureObject_t tree_xyzr,
								   cudaTextureObject_t tree_mass,
								   const int* body_n,
								   int block_size);

__host__ void fcompute_heap_bh_stackless(int offset_n1, int points_count, int tree_size,
										 nbcoord_t* f,
										 cudaTextureObject_t tree_xyzr,
										 cudaTextureObject_t tree_mass,
										 const int* body_n,
										 int block_size);
//! a[i] = value
__host__ void fill_buffer(nbcoord_t* ptr, nbcoord_t v, int count);

//! a[i] += b[i]*c
__host__ void fmadd_inplace(nbcoord_t* a, const nbcoord_t* b, nbcoord_t c, int count);

//! a[i] += b[i]*c with correction
__host__ void fmadd_inplace_corr(nbcoord_t* a, nbcoord_t* corr, const nbcoord_t* b,
								 nbcoord_t c, int count);

//! a[i+aoff] = b[i+boff] + c[i+coff]*d
__host__ void fmadd(nbcoord_t* a, const nbcoord_t* b, const nbcoord_t* c,
					nbcoord_t d, int count);

//! @result = max( fabs(a[k]), k=[0...asize) )
__host__ void fmaxabs(const nbcoord_t* a, int asize, nbcoord_t& result);

#endif //NBODY_ENGINE_CUDA_IMPL_H
