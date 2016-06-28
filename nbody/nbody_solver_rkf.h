#ifndef NBODY_SOLVER_RKF_H
#define NBODY_SOLVER_RKF_H

#include "nbody_solver_rk_butcher.h"

class nbody_solver_rkf : public nbody_solver_rk_butcher
{
public:
	nbody_solver_rkf();
	~nbody_solver_rkf();
};

#endif // NBODY_SOLVER_RKF_H
