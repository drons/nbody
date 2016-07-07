#include "nbody_solver_rklc.h"

/*!
   \brief Butcher table for Runge-Kutta-Lobatto IIIC fourth-order method
*/
class nbody_butcher_table_rklc : public nbody_butcher_table
{
public:
	nbody_butcher_table_rklc();

	size_t get_steps() const;
	const nbcoord_t**get_a() const;
	const nbcoord_t*get_b1() const;
	const nbcoord_t*get_b2() const;
	const nbcoord_t*get_c() const;
	bool is_implicit() const;
	bool is_embedded() const;
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
	static const nbcoord_t	a1[] = { 1.0/6.0, -1.0/3.0, 1.0/6.0 };
	static const nbcoord_t	a2[] = { 1.0/6.0, 5.0/12.0, -1.0/12.0 };
	static const nbcoord_t	a3[] = { 1.0/6.0, 2.0/3.0, 1.0/6.0 };
	static const nbcoord_t*	a[] = { a1, a2, a3 };

	return a;
}

const nbcoord_t* nbody_butcher_table_rklc::get_b1() const
{
	static const nbcoord_t	b1[] = { 1.0/6.0, 2.0/3.0, 1.0/6.0 };

	return b1;
}

const nbcoord_t* nbody_butcher_table_rklc::get_b2() const
{
	static const nbcoord_t	b2[] = { -1.0/2.0, 2.0, -1.0/2.0 };

	return b2;
}

const nbcoord_t*nbody_butcher_table_rklc::get_c() const
{
	static const nbcoord_t	c[]  = { 0, 0.5, 1.0 };

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
	nbody_solver_rk_butcher( new nbody_butcher_table_rklc() )
{

}
