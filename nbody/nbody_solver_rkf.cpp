#include "nbody_solver_rkf.h"
#include <QDebug>

/*!
   \brief Butcher table for Runge-Kutta-Fehlberg order 7 method
*/
class nbody_butcher_table_rkf : public nbody_butcher_table
{
public:
	nbody_butcher_table_rkf();

	size_t get_steps() const override;
	const nbcoord_t** get_a() const override;
	const nbcoord_t* get_b1() const override;
	const nbcoord_t* get_b2() const override;
	const nbcoord_t* get_c() const override;
	bool is_implicit() const override;
	bool is_embedded() const override;
};

nbody_butcher_table_rkf::nbody_butcher_table_rkf()
{
}

size_t nbody_butcher_table_rkf::get_steps() const
{
	return 13;
}

const nbcoord_t** nbody_butcher_table_rkf::get_a() const
{
	static const nbcoord_t	a1[] = { 0 };
	static const nbcoord_t	a2[] = { 2_f / 27_f };
	static const nbcoord_t	a3[] = { 1_f / 36_f, 1_f / 12_f};
	static const nbcoord_t	a4[] = { 1_f / 24_f, 0_f, 1_f / 8_f };
	static const nbcoord_t	a5[] = { 5_f / 12_f, 0_f, -25_f / 16_f, 25_f / 16_f };
	static const nbcoord_t	a6[] = { 1_f / 20_f, 0_f, 0_f, 1_f / 4_f, 1_f / 5_f };
	static const nbcoord_t	a7[] = { -25_f / 108_f, 0_f, 0_f, 125_f / 108_f, -65_f / 27_f, 125_f / 54_f };
	static const nbcoord_t	a8[] = { 31_f / 300_f, 0_f, 0_f, 0_f, 61_f / 225_f, -2_f / 9_f, 13_f / 900_f };
	static const nbcoord_t	a9[] = { 2_f, 0_f, 0_f, -53_f / 6_f, 704_f / 45_f, -107_f / 9_f, 67_f / 90_f, 3_f };
	static const nbcoord_t	a10[] = { -91_f / 108_f, 0_f, 0_f, 23_f / 108_f, -976_f / 135_f, 311_f / 54_f, -19 / 60_f, 17_f / 6_f, -1_f / 12_f };
	static const nbcoord_t	a11[] = { 2383_f / 4100_f, 0_f, 0_f, -341_f / 164_f, 4496_f / 1025_f, -301_f / 82_f, 2133_f / 4100_f, 45_f / 82_f, 45_f / 164_f, 18_f / 41_f };
	static const nbcoord_t	a12[] = { 3_f / 205_f, 0_f, 0_f, 0_f, 0_f, -6_f / 41_f, -3_f / 205_f, -3_f / 41_f, 3 / 41_f, 6_f / 41_f, 0_f };

	static const nbcoord_t	a13[] = { -1777_f / 4100_f, 0_f, 0_f, -341_f / 164_f, 4496_f / 1025_f, -289_f / 82_f, 2193_f / 4100_f, 51_f / 82_f, 33_f / 164_f, 19_f / 41_f, 0_f, 1_f };

	static const nbcoord_t*	a[] = { a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 };

	return a;
}

const nbcoord_t* nbody_butcher_table_rkf::get_b1() const
{
	static const nbcoord_t	b1[] = { 41_f / 840_f, 0_f, 0_f, 0_f, 0_f, 34_f / 105_f, 9_f / 35_f, 9_f / 35_f, 9_f / 280_f, 9_f / 280_f, 41_f / 840_f, 0_f, 0_f };

	return b1;
}

const nbcoord_t* nbody_butcher_table_rkf::get_b2() const
{
	static const nbcoord_t	b2[] = { 0_f, 0_f, 0_f, 0_f, 0_f, 34_f / 105_f, 9_f / 35_f, 9_f / 35_f, 9_f / 280_f, 9_f / 280_f, 0_f, 41_f / 840_f, 41_f / 840_f };

	return b2;
}

const nbcoord_t* nbody_butcher_table_rkf::get_c() const
{
	static const nbcoord_t	c[]  = { 0, 2_f / 27_f, 1_f / 9_f, 1_f / 6_f, 5_f / 12_f, 1_f / 2_f, 5_f / 6_f, 1_f / 6_f, 2_f / 3_f, 1_f / 3_f, 1_f, 0, 1_f };

	return c;
}

bool nbody_butcher_table_rkf::is_implicit() const
{
	return false;
}

bool nbody_butcher_table_rkf::is_embedded() const
{
	return true;
}

nbody_solver_rkf::nbody_solver_rkf() :
	nbody_solver_rk_butcher(new nbody_butcher_table_rkf)
{
}

nbody_solver_rkf::~nbody_solver_rkf()
{
}

const char* nbody_solver_rkf::type_name() const
{
	return "nbody_solver_rkf";
}

