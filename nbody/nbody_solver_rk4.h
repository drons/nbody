#ifndef NBODY_SOLVER_RK4_H
#define NBODY_SOLVER_RK4_H

#include "nbody_solver.h"

class nbody_solver_rk4 : public nbody_solver
{
	nbody_engine::memory*	m_k;
	nbody_engine::memory*	m_tmp;
	nbody_engine::memory*	m_coeff;
public:
	nbody_solver_rk4();
	~nbody_solver_rk4();
	const char* type_name() const;
	virtual void step( double dt );
};

#endif // NBODY_SOLVER_RK4_H
