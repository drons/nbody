#ifndef NBODY_SOLVER_RKFEAGIN12_H
#define NBODY_SOLVER_RKFEAGIN12_H

#include "nbody_solver_rk_butcher.h"

/*!
	\brief Feagin T. RK12(10) - a 12th-order method with an embedded 10th-order method

	@see https://sce.uhcl.edu/rungekutta/rk1210.txt

*/
class NBODY_DLL nbody_solver_rkfeagin12 : public nbody_solver_rk_butcher
{
public:
	nbody_solver_rkfeagin12();
	~nbody_solver_rkfeagin12();
	const char* type_name() const override;
};

#endif // NBODY_SOLVER_RKFEAGIN12_H
