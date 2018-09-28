#ifndef NBODY_SOLVER_RKCK_H
#define NBODY_SOLVER_RKCK_H

#include "nbody_solver_rk_butcher.h"

class nbody_solver_rkck : public nbody_solver_rk_butcher
{
public:
	nbody_solver_rkck();
	~nbody_solver_rkck();
	const char* type_name() const override;
};

#endif // NBODY_SOLVER_RKCK_H
