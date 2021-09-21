#ifndef NBODY_ENGINE_OPENCL_BH_H
#define NBODY_ENGINE_OPENCL_BH_H

#include "nbody_engine_opencl.h"
#include "nbody_engine_simple_bh.h"


class NBODY_DLL nbody_engine_opencl_bh : public nbody_engine_opencl
{
	nbcoord_t			m_distance_to_node_radius_ratio;
	size_t				m_tree_build_rate;
	e_traverse_type		m_traverse_type;
	e_tree_layout		m_tree_layout;
public:
	explicit nbody_engine_opencl_bh(nbcoord_t distance_to_node_radius_ratio,
									e_traverse_type tt, size_t tree_build_rate,
									e_tree_layout tl);
	~nbody_engine_opencl_bh();
	const char* type_name() const override;

	void fcompute(const nbcoord_t& t, const memory* y, memory* f) override;

	void print_info() const override;
};

#endif //NBODY_ENGINE_OPENCL_BH_H
