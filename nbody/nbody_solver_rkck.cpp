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
	static const nbcoord_t	a2[] = { 1_f / 5_f };
	static const nbcoord_t	a3[] = { 3_f / 40_f, 9_f / 40_f};
	static const nbcoord_t	a4[] = { 3_f / 10_f, -9_f / 10_f, 6_f / 5_f };
	static const nbcoord_t	a5[] = { -11_f / 54_f, 5_f / 2_f, -70_f / 27_f, 35_f / 27_f };
	static const nbcoord_t	a6[] = { 1631_f / 55296_f, 175_f / 512_f, 575_f / 13824_f, 44275_f / 110592_f, 253_f / 4096_f };
	static const nbcoord_t*	a[] = { a1, a2, a3, a4, a5, a6 };

	return a;
}

const nbcoord_t* nbody_butcher_table_rkck::get_b1() const
{
	static const nbcoord_t	b1[] = { 35_f / 378_f, 0_f, 250_f / 621_f, 125_f / 594_f, 0, 512_f / 1771_f };

	return b1;
}

const nbcoord_t* nbody_butcher_table_rkck::get_b2() const
{
	static const nbcoord_t	b2[] = { 2825_f / 27648_f, 0_f, 18575_f / 48384_f, 13525_f / 55296_f, 277_f / 14336_f, 1_f / 4_f };

	return b2;
}

const nbcoord_t* nbody_butcher_table_rkck::get_c() const
{
	static const nbcoord_t	c[]  = { 0, 1_f / 5_f, 3_f / 10_f, 3_f / 5_f, 1_f, 7_f / 8_f };

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
