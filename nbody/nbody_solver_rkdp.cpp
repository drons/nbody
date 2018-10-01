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
	static const nbcoord_t	a2[] = { 1.0 / 5.0 };
	static const nbcoord_t	a3[] = { 3.0 / 40.0, 9.0 / 40.0};
	static const nbcoord_t	a4[] = { 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0 };
	static const nbcoord_t	a5[] = { 19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0 };
	static const nbcoord_t	a6[] = { 9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0 };
	static const nbcoord_t	a7[] = { 35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0 };
	static const nbcoord_t*	a[] = { a1, a2, a3, a4, a5, a6, a7 };

	return a;
}

const nbcoord_t* nbody_butcher_table_rkdp::get_b1() const
{
	static const nbcoord_t	b1[] = { 35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0 };

	return b1;
}

const nbcoord_t* nbody_butcher_table_rkdp::get_b2() const
{
	static const nbcoord_t	b2[] = { 5179.0 / 57600.0, 0.0, 7571.0 / 16695.0, 393.0 / 640.0, -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0 };

	return b2;
}

const nbcoord_t* nbody_butcher_table_rkdp::get_c() const
{
	static const nbcoord_t	c[]  = { 0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0 };

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

