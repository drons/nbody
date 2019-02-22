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
	static const nbcoord_t	a2[] = { 1_f / 6_f };
	static const nbcoord_t	a3[] = { 4_f / 75_f, 16_f / 75_f};
	static const nbcoord_t	a4[] = { 5_f / 6_f, -8_f / 3_f, 5_f / 2_f };
	static const nbcoord_t	a5[] = { -165_f / 64_f, 55_f / 6_f, -425_f / 64_f, 85_f / 96_f };
	static const nbcoord_t	a6[] = { 12_f / 5_f, -8_f, 4015_f / 612_f, -11_f / 36_f, 88_f / 255_f };
	static const nbcoord_t	a7[] = { -8263_f / 15000_f, 124_f / 75_f, -643_f / 680_f, -81_f / 250_f, 2484_f / 10625_f, 0 };
	static const nbcoord_t	a8[] = { 3501_f / 1720_f, -300_f / 43_f, 297275_f / 52632_f, -319_f / 2322_f, 24068_f / 84065_f, 0_f, 3850_f / 26703_f  };
	static const nbcoord_t*	a[] = { a1, a2, a3, a4, a5, a6, a7, a8 };

	return a;
}

const nbcoord_t* nbody_butcher_table_rkdverk::get_b1() const
{
	static const nbcoord_t	b1[] = { 3_f / 40_f, 0_f, 875_f / 2244_f, 23_f / 72_f, 264_f / 1955_f, 0, 125_f / 11592_f, 43_f / 616_f };

	return b1;
}

const nbcoord_t* nbody_butcher_table_rkdverk::get_b2() const
{
	static const nbcoord_t	b2[] = { 13_f / 160_f, 0_f, 2375_f / 5984_f, 5_f / 16_f, 12_f / 85_f, 3_f / 44_f, 0_f, 0_f };

	return b2;
}

const nbcoord_t* nbody_butcher_table_rkdverk::get_c() const
{
	static const nbcoord_t	c[]  = { 0, 1_f / 6_f, 4_f / 15_f, 2_f / 3_f, 5_f / 6_f, 1_f, 1_f / 15_f, 1_f };

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

