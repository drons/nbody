#ifndef NBODY_SOLVER_EULER_H
#define NBODY_SOLVER_EULER_H

#include "nbody_solver.h"

class nbody_solver_euler : public nbody_solver
{
	nbody_engine::memory*	m_dy;
public:
	nbody_solver_euler();
	~nbody_solver_euler();
	virtual void step( nbcoord_t dt );
};

#endif // NBODY_SOLVER_EULER_H
