#ifndef NBODY_SOLVER_RKCK_H
#define NBODY_SOLVER_RKCK_H

#include "nbody_solver_rk_butcher.h"

class nbody_solver_rkck : public nbody_solver_rk_butcher
{
public:
	nbody_solver_rkck();
	~nbody_solver_rkck();
};

#endif // NBODY_SOLVER_RKCK_H
