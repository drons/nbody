#ifndef NBODY_SOLVER_TRAPEZE_H
#define NBODY_SOLVER_TRAPEZE_H

#include "nbody_solver.h"

class nbody_solver_trapeze : public nbody_solver
{
	nbody_engine::memory*	m_f01;
	nbody_engine::memory*	m_predictor;
	nbody_engine::memory*	m_coeff;
public:
	nbody_solver_trapeze();
	~nbody_solver_trapeze();
	virtual void step( nbcoord_t dt );
};

#endif // NBODY_SOLVER_TRAPEZE_H
