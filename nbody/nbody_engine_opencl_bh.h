#ifndef NBODY_ENGINE_OPENCL_BH_H
#define NBODY_ENGINE_OPENCL_BH_H

#include "nbody_engine_opencl.h"

class NBODY_DLL nbody_engine_opencl_bh : public nbody_engine_opencl
{
public:
	nbody_engine_opencl_bh();
	~nbody_engine_opencl_bh();
	virtual const char* type_name() const override;

	virtual void fcompute(const nbcoord_t& t, const memory* y, memory* f) override;
};

#endif //NBODY_ENGINE_OPENCL_BH_H
