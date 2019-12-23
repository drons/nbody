#include "nbody_butcher_table.h"

nbody_butcher_table::nbody_butcher_table()
{
}

nbody_butcher_table::~nbody_butcher_table()
{
}

nbody_butcher_table_rk4::nbody_butcher_table_rk4()
{
}

size_t nbody_butcher_table_rk4::get_steps() const
{
	return 4;
}

const nbcoord_t** nbody_butcher_table_rk4::get_a() const
{
	static const nbcoord_t	a1[] = { 0.0, 0.0 };
	static const nbcoord_t	a2[] = { 1.0 / 2.0, 0.0 };
	static const nbcoord_t	a3[] = { 0.0, 1.0 / 2.0, 0.0 };
	static const nbcoord_t	a4[] = { 0.0, 0.0, 1.0, 0.0 };
	static const nbcoord_t*	a[] = { a1, a2, a3, a4 };

	return a;
}

const nbcoord_t* nbody_butcher_table_rk4::get_b1() const
{
	static const nbcoord_t	b1[] = { 1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0 };

	return b1;
}

const nbcoord_t* nbody_butcher_table_rk4::get_b2() const
{
	return get_b1();
}

const nbcoord_t* nbody_butcher_table_rk4::get_c() const
{
	static const nbcoord_t	c[]  = { 0, 1_f / 2_f, 1_f / 2_f, 1_f };

	return c;
}

bool nbody_butcher_table_rk4::is_implicit() const
{
	return false;
}

bool nbody_butcher_table_rk4::is_embedded() const
{
	return false;
}
