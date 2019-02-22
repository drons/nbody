#include "nbody_solver_rklc.h"

/*!
   \brief Butcher table for Runge-Kutta-Lobatto IIIC fourth-order method
*/
class nbody_butcher_table_rklc : public nbody_butcher_table
{
public:
	nbody_butcher_table_rklc();

	size_t get_steps() const override;
	const nbcoord_t** get_a() const override;
	const nbcoord_t* get_b1() const override;
	const nbcoord_t* get_b2() const override;
	const nbcoord_t* get_c() const override;
	bool is_implicit() const override;
	bool is_embedded() const override;
};

nbody_butcher_table_rklc::nbody_butcher_table_rklc()
{

}

size_t nbody_butcher_table_rklc::get_steps() const
{
	return 3;
}

const nbcoord_t** nbody_butcher_table_rklc::get_a() const
{
	static const nbcoord_t	a1[] = { 1_f / 6_f, -1_f / 3_f, 1_f / 6_f };
	static const nbcoord_t	a2[] = { 1_f / 6_f, 5_f / 12_f, -1_f / 12_f };
	static const nbcoord_t	a3[] = { 1_f / 6_f, 2_f / 3_f, 1_f / 6_f };
	static const nbcoord_t*	a[] = { a1, a2, a3 };

	return a;
}

const nbcoord_t* nbody_butcher_table_rklc::get_b1() const
{
	static const nbcoord_t	b1[] = { 1_f / 6_f, 2_f / 3_f, 1_f / 6_f };

	return b1;
}

const nbcoord_t* nbody_butcher_table_rklc::get_b2() const
{
	static const nbcoord_t	b2[] = { -1_f / 2_f, 2_f, -1_f / 2_f };

	return b2;
}

const nbcoord_t* nbody_butcher_table_rklc::get_c() const
{
	static const nbcoord_t	c[]  = { 0, 1_f / 2_f, 1_f };

	return c;
}

bool nbody_butcher_table_rklc::is_implicit() const
{
	return true;
}

bool nbody_butcher_table_rklc::is_embedded() const
{
	return true;
}

nbody_solver_rklc::nbody_solver_rklc() :
	nbody_solver_rk_butcher(new nbody_butcher_table_rklc())
{

}

const char* nbody_solver_rklc::type_name() const
{
	return "nbody_solver_rklc";
}

