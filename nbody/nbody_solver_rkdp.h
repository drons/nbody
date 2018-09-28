#ifndef NBODY_SOLVER_RKDP_H
#define NBODY_SOLVER_RKDP_H

#include "nbody_solver_rk_butcher.h"

class nbody_solver_rkdp : public nbody_solver_rk_butcher
{
public:
	nbody_solver_rkdp();
	~nbody_solver_rkdp();
	const char* type_name() const override;
};

#endif // NBODY_SOLVER_RKDP_H
