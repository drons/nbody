#ifndef NBODY_ENGINE_OPENCL_BH_H
#define NBODY_ENGINE_OPENCL_BH_H

#include "nbody_engine_opencl.h"

class NBODY_DLL nbody_engine_opencl_bh : public nbody_engine_opencl
{
	nbcoord_t	m_distance_to_node_radius_ratio;
public:
	nbody_engine_opencl_bh(nbcoord_t distance_to_node_radius_ratio);
	~nbody_engine_opencl_bh();
	virtual const char* type_name() const override;

	virtual void fcompute(const nbcoord_t& t, const memory* y, memory* f) override;

	virtual void print_info() const override;
};

#endif //NBODY_ENGINE_OPENCL_BH_H
