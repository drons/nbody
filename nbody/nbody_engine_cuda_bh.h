#ifndef NBODY_ENGINE_CUDA_BH_H
#define NBODY_ENGINE_CUDA_BH_H

#include "nbody_engine_cuda.h"
#include "nbody_space_heap.h"

class NBODY_DLL nbody_engine_cuda_bh : public nbody_engine_cuda
{
	bool		m_cycle_traverse;
	nbcoord_t	m_distance_to_node_radius_ratio;
	size_t		m_tree_build_rate;

	nbody_space_heap	m_heap;

	smemory*		m_dev_tree_cmx;
	smemory*		m_dev_tree_cmy;
	smemory*		m_dev_tree_cmz;
	smemory*		m_dev_bmin_cmx;
	smemory*		m_dev_bmin_cmy;
	smemory*		m_dev_bmin_cmz;
	smemory*		m_dev_bmax_cmx;
	smemory*		m_dev_bmax_cmy;
	smemory*		m_dev_bmax_cmz;
	smemory*		m_dev_tree_mass;
	smemory*		m_dev_tree_crit_r2;
	smemory*		m_dev_indites;

public:
	explicit nbody_engine_cuda_bh(nbcoord_t distance_to_node_radius_ratio,
								  size_t tree_build_rate);
	~nbody_engine_cuda_bh();
	const char* type_name() const override;
	void fcompute(const nbcoord_t& t, const memory* y, memory* f) override;
	void print_info() const override;
};

#endif // NBODY_ENGINE_CUDA_BH_H
