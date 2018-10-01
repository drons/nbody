#include "nbody_solver_rkck.h"
#include <QDebug>

/*!
   \brief Butcher table for Runge-Kutta-Cash–Karp order 5 method
*/
class nbody_butcher_table_rkck : public nbody_butcher_table
{
public:
	nbody_butcher_table_rkck();

	size_t get_steps() const override;
	const nbcoord_t** get_a() const override;
	const nbcoord_t* get_b1() const override;
	const nbcoord_t* get_b2() const override;
	const nbcoord_t* get_c() const override;
	bool is_implicit() const override;
	bool is_embedded() const override;
};

nbody_butcher_table_rkck::nbody_butcher_table_rkck()
{

}

size_t nbody_butcher_table_rkck::get_steps() const
{
	return 6;
}

const nbcoord_t** nbody_butcher_table_rkck::get_a() const
{
	static const nbcoord_t	a1[] = { 0 };
	static const nbcoord_t	a2[] = { 1.0 / 5.0 };
	static const nbcoord_t	a3[] = { 3.0 / 40.0, 9.0 / 40.0};
	static const nbcoord_t	a4[] = { 3.0 / 10.0, -9.0 / 10.0, 6.0 / 5.0 };
	static const nbcoord_t	a5[] = { -11.0 / 54.0, 5.0 / 2.0, -70.0 / 27.0, 35.0 / 27.0 };
	static const nbcoord_t	a6[] = { 1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 / 110592.0, 253.0 / 4096.0 };
	static const nbcoord_t*	a[] = { a1, a2, a3, a4, a5, a6 };

	return a;
}

const nbcoord_t* nbody_butcher_table_rkck::get_b1() const
{
	static const nbcoord_t	b1[] = { 35.0 / 378.0, 0.0, 250.0 / 621.0, 125.0 / 594.0, 0, 512.0 / 1771.0 };

	return b1;
}

const nbcoord_t* nbody_butcher_table_rkck::get_b2() const
{
	static const nbcoord_t	b2[] = { 2825.0 / 27648.0, 0.0, 18575.0 / 48384.0, 13525.0 / 55296.0, 277.0 / 14336.0, 1.0 / 4.0 };

	return b2;
}

const nbcoord_t* nbody_butcher_table_rkck::get_c() const
{
	static const nbcoord_t	c[]  = { 0, 1.0 / 5.0, 3.0 / 10.0, 3.0 / 5.0, 1.0, 7.0 / 8.0 };

	return c;
}

bool nbody_butcher_table_rkck::is_implicit() const
{
	return false;
}

bool nbody_butcher_table_rkck::is_embedded() const
{
	return true;
}

nbody_solver_rkck::nbody_solver_rkck() :
	nbody_solver_rk_butcher(new nbody_butcher_table_rkck)
{
}

nbody_solver_rkck::~nbody_solver_rkck()
{
}

const char* nbody_solver_rkck::type_name() const
{
	return "nbody_solver_rkck";
}
