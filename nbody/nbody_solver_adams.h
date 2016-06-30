#ifndef NBODY_SOLVER_ADAMS_H
#define NBODY_SOLVER_ADAMS_H

#include "nbody_solver.h"

/*!
   \brief Adams–Bashforth method
 */
class nbody_solver_adams : public nbody_solver
{
	nbody_solver*			m_starter;
	nbody_engine::memory*	m_f;
	nbody_engine::memory*	m_coeff;
public:
	nbody_solver_adams();
	~nbody_solver_adams();
	void step(double dt);
};

#endif // NBODY_SOLVER_ADAMS_H
