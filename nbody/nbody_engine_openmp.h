#ifndef NBODY_ENGINE_OPENMP_H
#define NBODY_ENGINE_OPENMP_H

#include "nbody_engine_simple.h"

class NBODY_DLL nbody_engine_openmp : public nbody_engine_simple
{
public:
	nbody_engine_openmp();
	~nbody_engine_openmp();

	virtual const char* type_name() const override;

	virtual void fcompute(const nbcoord_t& t, const memory* y, memory* f) override;

	virtual void copy_buffer(memory* a, const memory* b) override;
	virtual void fill_buffer(memory* a, const nbcoord_t& value) override;
	virtual void fmadd_inplace(memory* a, const memory* b, const nbcoord_t& c) override;
	virtual void fmadd(memory* a, const memory* b, const memory* c, const nbcoord_t& d) override;
	virtual void fmaxabs(const memory* a, nbcoord_t& result) override;

	virtual void print_info() const override;
};

#endif // NBODY_ENGINE_SIMPLE_H
