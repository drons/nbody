#include "nbody_solver_rkdverk.h"
#include <QDebug>

/*!
   \brief Butcher table for Verner's method of order 6(5) (DVERK)
*/
class nbody_butcher_table_rkdverk : public nbody_butcher_table
{
public:
	nbody_butcher_table_rkdverk();

	size_t get_steps() const override;
	const nbcoord_t** get_a() const override;
	const nbcoord_t* get_b1() const override;
	const nbcoord_t* get_b2() const override;
	const nbcoord_t* get_c() const override;
	bool is_implicit() const override;
	bool is_embedded() const override;
};

nbody_butcher_table_rkdverk::nbody_butcher_table_rkdverk()
{

}

size_t nbody_butcher_table_rkdverk::get_steps() const
{
	return 8;
}

const nbcoord_t** nbody_butcher_table_rkdverk::get_a() const
{
	static const nbcoord_t	a1[] = { 0 };
	static const nbcoord_t	a2[] = { 1.0 / 6.0 };
	static const nbcoord_t	a3[] = { 4.0 / 75.0, 16.0 / 75.0};
	static const nbcoord_t	a4[] = { 5.0 / 6.0, -8.0 / 3.0, 5.0 / 2.0 };
	static const nbcoord_t	a5[] = { -165.0 / 64.0, 55.0 / 6.0, -425.0 / 64.0, 85.0 / 96.0 };
	static const nbcoord_t	a6[] = { 12.0 / 5.0, -8.0, 4015.0 / 612.0, -11.0 / 36.0, 88.0 / 255.0 };
	static const nbcoord_t	a7[] = { -8263.0 / 15000.0, 124.0 / 75.0, -643.0 / 680.0, -81.0 / 250.0, 2484.0 / 10625.0, 0 };
	static const nbcoord_t	a8[] = { 3501.0 / 1720.0, -300.0 / 43.0, 297275.0 / 52632.0, -319.0 / 2322.0, 24068.0 / 84065.0, 0.0, 3850.0 / 26703.0  };
	static const nbcoord_t*	a[] = { a1, a2, a3, a4, a5, a6, a7, a8 };

	return a;
}

const nbcoord_t* nbody_butcher_table_rkdverk::get_b1() const
{
	static const nbcoord_t	b1[] = { 3.0 / 40.0, 0.0, 875.0 / 2244.0, 23.0 / 72.0, 264.0 / 1955.0, 0, 125.0 / 11592.0, 43.0 / 616.0 };

	return b1;
}

const nbcoord_t* nbody_butcher_table_rkdverk::get_b2() const
{
	static const nbcoord_t	b2[] = { 13.0 / 160.0, 0.0, 2375.0 / 5984.0, 5.0 / 16.0, 12.0 / 85.0, 3.0 / 44.0, 0.0, 0.0 };

	return b2;
}

const nbcoord_t* nbody_butcher_table_rkdverk::get_c() const
{
	static const nbcoord_t	c[]  = { 0, 1.0 / 6.0, 4.0 / 15.0, 2.0 / 3.0, 5.0 / 6.0, 1.0, 1.0 / 15.0, 1.0 };

	return c;
}

bool nbody_butcher_table_rkdverk::is_implicit() const
{
	return false;
}

bool nbody_butcher_table_rkdverk::is_embedded() const
{
	return true;
}

nbody_solver_rkdverk::nbody_solver_rkdverk() :
	nbody_solver_rk_butcher(new nbody_butcher_table_rkdverk)
{
}

nbody_solver_rkdverk::~nbody_solver_rkdverk()
{
}

const char* nbody_solver_rkdverk::type_name() const
{
	return "nbody_solver_rkdverk";
}

