#ifndef NBODY_SOLVER_RKGL_H
#define NBODY_SOLVER_RKGL_H

#include "nbody_solver_rk_butcher.h"

/*!
   \brief Gauss–Legendre 6-order method
*/
class nbody_solver_rkgl : public nbody_solver_rk_butcher
{
public:
	nbody_solver_rkgl( nbody_data* data );
};

#endif // NBODY_SOLVER_RKGL_H
