#include "nbody_solver_rkf.h"
#include <QDebug>

/*!
   \brief Butcher table for Runge-Kutta-Fehlberg order 7 method
*/
class nbody_butcher_table_rkf : public nbody_butcher_table
{
public:
	nbody_butcher_table_rkf();

	size_t get_steps() const;
	const nbcoord_t**get_a() const;
	const nbcoord_t*get_b1() const;
	const nbcoord_t*get_b2() const;
	const nbcoord_t*get_c() const;
	bool is_implicit() const;
	bool is_embedded() const;
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
	static const nbcoord_t	a2[] = { 2.0/27.0 };
	static const nbcoord_t	a3[] = { 1.0/36.0, 1.0/12.0};
	static const nbcoord_t	a4[] = { 1.0/24.0, 0.0, 1.0/8.0 };
	static const nbcoord_t	a5[] = { 5.0/12.0, 0.0, -25.0/16.0, 25.0/16.0 };
	static const nbcoord_t	a6[] = { 1.0/20.0, 0.0, 0.0, 1.0/4.0, 1.0/5.0 };
	static const nbcoord_t	a7[] = { -25.0/108.0, 0.0, 0.0, 125.0/108.0, -65.0/27.0, 125.0/54.0 };
	static const nbcoord_t	a8[] = { 31.0/300.0, 0.0, 0.0, 0.0, 61.0/225.0, -2.0/9.0, 13.0/900.0 };
	static const nbcoord_t	a9[] = { 2.0, 0.0, 0.0, -53.0/6.0, 704.0/45.0, -107.0/9.0, 67.0/90.0, 3.0 };
	static const nbcoord_t	a10[] = { -91.0/108.0, 0.0, 0.0, 23.0/108.0, -976.0/135.0, 311.0/54.0, -19/60.0, 17.0/6.0, -1.0/12.0 };
	static const nbcoord_t	a11[] = { 2383.0/4100.0, 0.0, 0.0, -341.0/164.0, 4496.0/1025.0, -301.0/82.0, 2133.0/4100.0, 45.0/82.0, 45.0/164.0, 18.0/41.0 };
	static const nbcoord_t	a12[] = { 3.0/205.0, 0.0, 0.0, 0.0, 0.0, -6.0/41.0, -3.0/205.0, -3.0/41.0, 3/41.0, 6.0/41.0, 0.0 };

	static const nbcoord_t	a13[] = { -1777.0/4100.0, 0.0, 0.0, -341.0/164.0, 4496.0/1025.0, -289.0/82.0, 2193.0/4100.0, 51.0/82.0, 33.0/164.0, 19.0/41.0, 0.0, 1.0 };

	static const nbcoord_t*	a[] = { a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13 };

	return a;
}

const nbcoord_t* nbody_butcher_table_rkf::get_b1() const
{
	static const nbcoord_t	b1[] = { 41.0/840.0, 0.0, 0.0, 0.0, 0.0, 34.0/105.0, 9.0/35.0, 9.0/35.0, 9.0/280.0, 9.0/280.0, 41.0/840.0, 0.0, 0.0 };

	return b1;
}

const nbcoord_t* nbody_butcher_table_rkf::get_b2() const
{
	static const nbcoord_t	b2[] = { 0.0, 0.0, 0.0, 0.0, 0.0, 34.0/105.0, 9.0/35.0, 9.0/35.0, 9.0/280.0, 9.0/280.0, 0.0, 41.0/840.0, 41.0/840.0 };

	return b2;
}

const nbcoord_t*nbody_butcher_table_rkf::get_c() const
{
	static const nbcoord_t	c[]  = { 0, 2.0/27.0, 1.0/9.0, 1.0/6.0, 5.0/12.0, 1.0/2.0, 5.0/6.0, 1.0/6.0, 2.0/3.0, 1.0/3.0, 1.0, 0, 1.0 };

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

nbody_solver_rkf::nbody_solver_rkf( nbody_data* data ) :
	nbody_solver_rk_butcher( data, new nbody_butcher_table_rkf )
{
}

nbody_solver_rkf::~nbody_solver_rkf()
{
}
