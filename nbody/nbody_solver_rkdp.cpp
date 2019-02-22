#include "nbody_solver_rkdp.h"
#include <QDebug>

/*!
   \brief Butcher table for classic Runge-Kutta-Dormand–Prince order 5 method
*/
class nbody_butcher_table_rkdp : public nbody_butcher_table
{
public:
	nbody_butcher_table_rkdp();

	size_t get_steps() const override;
	const nbcoord_t** get_a() const override;
	const nbcoord_t* get_b1() const override;
	const nbcoord_t* get_b2() const override;
	const nbcoord_t* get_c() const override;
	bool is_implicit() const override;
	bool is_embedded() const override;
};

nbody_butcher_table_rkdp::nbody_butcher_table_rkdp()
{

}

size_t nbody_butcher_table_rkdp::get_steps() const
{
	return 7;
}

const nbcoord_t** nbody_butcher_table_rkdp::get_a() const
{
	static const nbcoord_t	a1[] = { 0 };
	static const nbcoord_t	a2[] = { 1_f / 5_f };
	static const nbcoord_t	a3[] = { 3_f / 40_f, 9_f / 40_f};
	static const nbcoord_t	a4[] = { 44_f / 45_f, -56_f / 15_f, 32_f / 9_f };
	static const nbcoord_t	a5[] = { 19372_f / 6561_f, -25360_f / 2187_f, 64448_f / 6561_f, -212_f / 729_f };
	static const nbcoord_t	a6[] = { 9017_f / 3168_f, -355_f / 33_f, 46732_f / 5247_f, 49_f / 176_f, -5103_f / 18656_f };
	static const nbcoord_t	a7[] = { 35_f / 384_f, 0_f, 500_f / 1113_f, 125_f / 192_f, -2187_f / 6784_f, 11_f / 84_f, 0_f };
	static const nbcoord_t*	a[] = { a1, a2, a3, a4, a5, a6, a7 };

	return a;
}

const nbcoord_t* nbody_butcher_table_rkdp::get_b1() const
{
	static const nbcoord_t	b1[] = { 35_f / 384_f, 0_f, 500_f / 1113_f, 125_f / 192_f, -2187_f / 6784_f, 11_f / 84_f, 0_f };

	return b1;
}

const nbcoord_t* nbody_butcher_table_rkdp::get_b2() const
{
	static const nbcoord_t	b2[] = { 5179_f / 57600_f, 0_f, 7571_f / 16695_f, 393_f / 640_f, -92097_f / 339200_f, 187_f / 2100_f, 1_f / 40_f };

	return b2;
}

const nbcoord_t* nbody_butcher_table_rkdp::get_c() const
{
	static const nbcoord_t	c[]  = { 0, 1_f / 5_f, 3_f / 10_f, 4_f / 5_f, 8_f / 9_f, 1_f, 1_f };

	return c;
}

bool nbody_butcher_table_rkdp::is_implicit() const
{
	return false;
}

bool nbody_butcher_table_rkdp::is_embedded() const
{
	return true;
}

nbody_solver_rkdp::nbody_solver_rkdp() :
	nbody_solver_rk_butcher(new nbody_butcher_table_rkdp)
{
}

nbody_solver_rkdp::~nbody_solver_rkdp()
{
}

const char* nbody_solver_rkdp::type_name() const
{
	return "nbody_solver_rkdp";
}

