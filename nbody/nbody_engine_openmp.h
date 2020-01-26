#ifndef NBODY_ENGINE_OPENMP_H
#define NBODY_ENGINE_OPENMP_H

#include "nbody_engine_simple.h"

class NBODY_DLL nbody_engine_openmp : public nbody_engine_simple
{
public:
	nbody_engine_openmp();
	~nbody_engine_openmp();

	const char* type_name() const override;

	void fcompute(const nbcoord_t& t, const memory* y, memory* f) override;

	void copy_buffer(memory* a, const memory* b) override;
	void fill_buffer(memory* a, const nbcoord_t& value) override;
	void fmadd_inplace(memory* a, const memory* b, const nbcoord_t& c) override;
	void fmadd(memory* a, const memory* b, const memory* c, const nbcoord_t& d) override;
	void fmaddn_corr(memory* a, memory* corr, const memory_array& b,
					 const nbcoord_t* c, size_t csize) override;
	void fmaxabs(const memory* a, nbcoord_t& result) override;

	void print_info() const override;
};

#endif // NBODY_ENGINE_SIMPLE_H
