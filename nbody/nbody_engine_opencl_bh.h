#ifndef NBODY_ENGINE_OPENCL_BH_H
#define NBODY_ENGINE_OPENCL_BH_H

#include "nbody_engine_opencl.h"

class NBODY_DLL nbody_engine_opencl_bh : public nbody_engine_opencl
{
	nbcoord_t	m_distance_to_node_radius_ratio;
	size_t		m_tree_build_rate;
	bool		m_cycle_traverse;
public:
	explicit nbody_engine_opencl_bh(nbcoord_t distance_to_node_radius_ratio,
									size_t tree_build_rate);
	~nbody_engine_opencl_bh();
	const char* type_name() const override;

	void fcompute(const nbcoord_t& t, const memory* y, memory* f) override;

	void print_info() const override;
	void set_cycle_traverse(bool tt);
};

#endif //NBODY_ENGINE_OPENCL_BH_H
