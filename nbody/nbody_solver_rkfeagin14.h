#ifndef NBODY_SOLVER_RKFEAGIN14_H
#define NBODY_SOLVER_RKFEAGIN14_H

#include "nbody_solver_rk_butcher.h"

/*!
	\brief Feagin T. RK14(12) - a 14th-order method with an embedded 12th-order method

	@see https://sce.uhcl.edu/rungekutta/rk1412.txt

*/
class NBODY_DLL nbody_solver_rkfeagin14 : public nbody_solver_rk_butcher
{
public:
	nbody_solver_rkfeagin14();
	~nbody_solver_rkfeagin14();
	const char* type_name() const override;
};

#endif // NBODY_SOLVER_RKFEAGIN14_H
