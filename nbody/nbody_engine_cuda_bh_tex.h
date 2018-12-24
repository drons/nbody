#ifndef NBODY_ENGINE_CUDA_BH_TEX_H
#define NBODY_ENGINE_CUDA_BH_TEX_H

#include "nbody_engine_cuda.h"

class NBODY_DLL nbody_engine_cuda_bh_tex : public nbody_engine_cuda
{
	bool		m_cycle_traverse;
	nbcoord_t	m_distance_to_node_radius_ratio;

	smemory*		m_dev_tree_cmx;
	smemory*		m_dev_tree_cmy;
	smemory*		m_dev_tree_cmz;
	smemory*		m_dev_tree_mass;
	smemory*		m_dev_tree_crit_r2;
	smemory*		m_dev_indites;

public:
	explicit nbody_engine_cuda_bh_tex(nbcoord_t distance_to_node_radius_ratio);
	~nbody_engine_cuda_bh_tex();
	virtual const char* type_name() const override;
	virtual void fcompute(const nbcoord_t& t, const memory* y, memory* f) override;
	virtual void print_info() const override;
};

#endif // NBODY_ENGINE_CUDA_BH_TEX_H
