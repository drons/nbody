#ifndef NBODY_SOLVER_RKF_H
#define NBODY_SOLVER_RKF_H

#include "nbody_solver_rk_butcher.h"

class NBODY_DLL nbody_solver_rkf : public nbody_solver_rk_butcher
{
public:
	nbody_solver_rkf();
	~nbody_solver_rkf();
	const char* type_name() const override;
};

#endif // NBODY_SOLVER_RKF_H
