#ifndef NBODY_SOLVER_RKFEAGIN10_H
#define NBODY_SOLVER_RKFEAGIN10_H

#include "nbody_solver_rk_butcher.h"

/*!
	\brief Runge-Kutta-Feagin order 10 method

	@see Feagin T. A tenth-order Runge–Kutta method with error estimate.
		 In: Proc. of the IAENG Conf. on Scientific Computing. Hong Kong, 2007.
		 https://sce.uhcl.edu/feagin/courses/rk10.pdf

*/
class NBODY_DLL nbody_solver_rkfeagin10 : public nbody_solver_rk_butcher
{
public:
	nbody_solver_rkfeagin10();
	~nbody_solver_rkfeagin10();
	const char* type_name() const override;
};

#endif // NBODY_SOLVER_RKFEAGIN10_H
