#ifndef NBODY_ENGINE_BLOCK_H
#define NBODY_ENGINE_BLOCK_H

#include "nbody_engine_openmp.h"

class NBODY_DLL nbody_engine_block : public nbody_engine_openmp
{
public:
	nbody_engine_block();
	virtual const char* type_name() const override;
	virtual void fcompute(const nbcoord_t& t, const memory* y, memory* f) override;
};

#endif // NBODY_ENGINE_BLOCK_H
