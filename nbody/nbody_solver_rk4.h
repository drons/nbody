#ifndef NBODY_SOLVER_RK4_H
#define NBODY_SOLVER_RK4_H

#include "nbody_solver.h"

class NBODY_DLL nbody_solver_rk4 : public nbody_solver
{
	nbody_engine::memory_array	m_k;
	nbody_engine::memory*		m_tmp;
public:
	nbody_solver_rk4();
	~nbody_solver_rk4();
	const char* type_name() const override;
	void advise(nbcoord_t dt) override;
};

#endif // NBODY_SOLVER_RK4_H
