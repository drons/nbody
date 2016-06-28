#ifndef NBODY_SOLVER_RKLC_H
#define NBODY_SOLVER_RKLC_H

#include "nbody_solver_rk_butcher.h"

/*!
   \brief Lobatto IIIC 4-order method
*/
class nbody_solver_rklc : public nbody_solver_rk_butcher
{
public:
	nbody_solver_rklc();
};

#endif // NBODY_SOLVER_RKLC_H
