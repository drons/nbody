#ifndef NBODY_SOLVER_RKDVERK_H
#define NBODY_SOLVER_RKDVERK_H

#include "nbody_solver_rk_butcher.h"

class NBODY_DLL nbody_solver_rkdverk : public nbody_solver_rk_butcher
{
public:
	nbody_solver_rkdverk();
	~nbody_solver_rkdverk();
	const char* type_name() const override;
};

#endif // NBODY_SOLVER_RKDVERK_H
