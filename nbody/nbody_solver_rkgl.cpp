#include "nbody_solver_rkgl.h"

static const nbcoord_t	sqrt15(sqrt(15.0));
/*!
   \brief Butcher table for Gauss–Legendre 6-order method
*/
class nbody_butcher_table_rkgl : public nbody_butcher_table
{
public:
	nbody_butcher_table_rkgl();

	size_t get_steps() const override;
	const nbcoord_t** get_a() const override;
	const nbcoord_t* get_b1() const override;
	const nbcoord_t* get_b2() const override;
	const nbcoord_t* get_c() const override;
	bool is_implicit() const override;
	bool is_embedded() const override;
};

nbody_butcher_table_rkgl::nbody_butcher_table_rkgl()
{

}

size_t nbody_butcher_table_rkgl::get_steps() const
{
	return 3;
}

const nbcoord_t** nbody_butcher_table_rkgl::get_a() const
{
	static const nbcoord_t	a1[] = { 15.0 / 36.0, 2.0 / 9.0 - sqrt15 / 15.0, 5.0 / 36.0 - sqrt15 / 30.0 };
	static const nbcoord_t	a2[] = { 5.0 / 36.0 + sqrt15 / 24.0, 2.0 / 9.0, 5.0 / 36.0 - sqrt15 / 24.0 };
	static const nbcoord_t	a3[] = { 5.0 / 36.0 + sqrt15 / 30.0, 2.0 / 9.0 + sqrt15 / 15.0, 5.0 / 36.0 };
	static const nbcoord_t*	a[] = { a1, a2, a3 };

	return a;
}

const nbcoord_t* nbody_butcher_table_rkgl::get_b1() const
{
	static const nbcoord_t	b1[] = { 5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0 };

	return b1;
}

const nbcoord_t* nbody_butcher_table_rkgl::get_b2() const
{
	static const nbcoord_t	b2[] = { -5.0 / 6.0, 8.0 / 3.0, -5.0 / 6.0 };

	return b2;
}

const nbcoord_t* nbody_butcher_table_rkgl::get_c() const
{
	static const nbcoord_t	c[]  = { 1.0 / 2.0 - sqrt15 / 10.0, 0.5, 1.0 / 2.0 + sqrt15 / 10.0 };

	return c;
}

bool nbody_butcher_table_rkgl::is_implicit() const
{
	return true;
}

bool nbody_butcher_table_rkgl::is_embedded() const
{
	return false;
}

nbody_solver_rkgl::nbody_solver_rkgl() :
	nbody_solver_rk_butcher(new nbody_butcher_table_rkgl())
{

}

const char* nbody_solver_rkgl::type_name() const
{
	return "nbody_solver_rkgl";
}

