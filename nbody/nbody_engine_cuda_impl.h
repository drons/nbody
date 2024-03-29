#ifndef NBODY_ENGINE_CUDA_IMPL_H
#define NBODY_ENGINE_CUDA_IMPL_H

#include <cuda_runtime.h>
#ifdef HAVE_NCCL
#include <nccl.h>
#endif //HAVE_NCCL
#include "nbtype.h"

void cuda_check(const char* file, int line, const char* context_name, cudaError_t res);
#define CUDACHECK(func) cuda_check(__FILE__, __LINE__, #func, func)

#ifdef HAVE_NCCL
void nccl_check(const char* file, int line, const char* context_name, ncclResult_t res);
#define NCCLCHECK(func) nccl_check(__FILE__, __LINE__, #func, func)
#endif //HAVE_NCCL


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

__host__ void fcompute_heap_bh_tex(int offset_n1, int points_count,
								   int compute_points_count, int tree_size,
								   const nbcoord_t* y,
								   nbcoord_t* f,
								   cudaTextureObject_t tree_xyzr,
								   cudaTextureObject_t tree_mass,
								   const int* body_n,
								   int block_size);

__host__ void fcompute_heap_bh_stackless(int offset_n1, int points_count,
										 int compute_points_count, int tree_size,
										 const nbcoord_t* y,
										 nbcoord_t* f,
										 cudaTextureObject_t tree_xyzr,
										 cudaTextureObject_t tree_mass,
										 const int* body_n,
										 int block_size);
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
							 const int* body_n);
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
							 nbcoord_t distance_to_node_radius_ratio_sqr);
__host__ void update_leaf_bh_tex(int points_count,
								 const nbcoord_t* y,
								 nbcoord_t* tree_xyzr,
								 nbcoord_t* bmin_cmx,
								 nbcoord_t* bmin_cmy,
								 nbcoord_t* bmin_cmz,
								 nbcoord_t* bmax_cmx,
								 nbcoord_t* bmax_cmy,
								 nbcoord_t* bmax_cmz,
								 const int* body_n);
__host__ void update_node_bh_tex(int level_size,
								 nbcoord_t* tree_xyzr,
								 nbcoord_t* bmin_cmx,
								 nbcoord_t* bmin_cmy,
								 nbcoord_t* bmin_cmz,
								 nbcoord_t* bmax_cmx,
								 nbcoord_t* bmax_cmy,
								 nbcoord_t* bmax_cmz,
								 nbcoord_t* tree_mass,
								 nbcoord_t distance_to_node_radius_ratio_sqr);
//! Clamp coord
__host__ void clamp_coord(nbcoord_t* y, nbcoord_t b, int count);

//! a[i] = value
__host__ void fill_buffer(nbcoord_t* ptr, nbcoord_t v, int count);

//! a[i] += b[i]*c
__host__ void fmadd_inplace(nbcoord_t* a, const nbcoord_t* b, nbcoord_t c, int count, cudaStream_t s);

//! a[i] += b[i]*c with correction
__host__ void fmadd_inplace_corr(nbcoord_t* a, nbcoord_t* corr, const nbcoord_t* b,
								 nbcoord_t c, int count);

//! a[i+aoff] = b[i+boff] + c[i+coff]*d
__host__ void fmadd(nbcoord_t* a, const nbcoord_t* b, const nbcoord_t* c,
					nbcoord_t d, int count);

//! @result = max( fabs(a[k]), k=[0...asize) )
__host__ void fmaxabs(const nbcoord_t* a, int asize, nbcoord_t& result);

#endif //NBODY_ENGINE_CUDA_IMPL_H
