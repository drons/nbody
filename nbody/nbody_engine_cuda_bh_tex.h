#ifndef NBODY_ENGINE_CUDA_BH_TEX_H
#define NBODY_ENGINE_CUDA_BH_TEX_H

#include "nbody_engine_cuda.h"
#include "nbody_engine_simple_bh.h"

class NBODY_DLL nbody_engine_cuda_bh_tex : public nbody_engine_cuda
{
	bool				m_cycle_traverse;
	e_tree_layout		m_tree_layout;
	nbcoord_t			m_distance_to_node_radius_ratio;
	size_t				m_tree_build_rate;

	nbody_space_heap	m_heap;

	smemory*			m_dev_tree_xyzr;
	smemory*			m_dev_bmin_cmx;
	smemory*			m_dev_bmin_cmy;
	smemory*			m_dev_bmin_cmz;
	smemory*			m_dev_bmax_cmx;
	smemory*			m_dev_bmax_cmy;
	smemory*			m_dev_bmax_cmz;
	smemory*			m_dev_tree_mass;
	smemory*			m_dev_indites;

public:
	explicit nbody_engine_cuda_bh_tex(nbcoord_t distance_to_node_radius_ratio,
									  size_t tree_build_rate,
									  e_tree_layout tl);
	~nbody_engine_cuda_bh_tex();
	const char* type_name() const override;
	void fcompute(const nbcoord_t& t, const memory* y, memory* f) override;
	void print_info() const override;
};

#endif // NBODY_ENGINE_CUDA_BH_TEX_H
