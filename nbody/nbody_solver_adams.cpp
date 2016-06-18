#include "nbody_solver_adams.h"
#include "nbody_solver_euler.h"
#include "summation.h"
#include <QDebug>

nbody_solver_adams::nbody_solver_adams( nbody_data* data ) : nbody_solver( data )
{
	m_starter = new nbody_solver_euler( data );
}

nbody_solver_adams::~nbody_solver_adams()
{
	delete m_starter;
}

void nbody_solver_adams::step( double dt )
{
	const size_t		rank = 5;
	const nbcoord_t		a1[1] = { 1.0 };
	const nbcoord_t		a2[2] = { 3.0/2.0, -1.0/2.0 };
	const nbcoord_t		a3[3] = { 23.0/12.0, -4.0/3.0, 5.0/12.0 };
	const nbcoord_t		a4[4] = { 55.0/24.0, -59.0/24.0, 37.0/24.0, -3.0/8.0 };
	const nbcoord_t		a5[5] = { 1901.0/720.0, -1387.0/360.0, 109.0/30.0, -637.0/360.0, 251.0/720.0 };
	const nbcoord_t*	ar[] = { NULL, a1, a2, a3, a4, a5 };
	const nbcoord_t*	a = ar[rank];
	nbvertex_t*			vertites = data()->get_vertites();
	nbvertex_t*			velosites = data()->get_velosites();
	size_t				count = data()->get_count();

	if( m_xdata.size() != rank )
	{
		m_starter->set_engine( get_engine() );
		m_xdata.resize(rank);
		m_vdata.resize(rank);
		m_dx.resize(rank);
		m_dv.resize(rank);
		m_correction_vert.resize( count );
		m_correction_vel.resize( count );

		for( size_t r = 0; r != rank; ++r )
		{
			m_xdata[r].resize(count);
			m_vdata[r].resize(count);
			m_dx[r] = m_xdata[r].data();
			m_dv[r] = m_vdata[r].data();
		}
	}

	if( data()->get_step() > rank )
	{
		step_v( vertites, m_dv.front() );

		nbvertex_t* dxfront = m_dx.front();

		//#pragma omp parallel for
		for( size_t n = 0; n < count; ++n )
		{
			nbvertex_t dx;
			dxfront[n] = velosites[n];

			for( size_t r = 0; r != rank; ++r )
			{
				dx += m_dx[r][n]*a[r];
			}
			vertites[n] = summation_k( vertites[n], dx*dt, &m_correction_vert[n] );
			//qDebug() << (dx - m_dx[0][n]).length()/dx.length() << velosites[n].length();
		}
		//#pragma omp parallel for
		for( size_t n = 0; n < count; ++n )
		{
			nbvertex_t dv;
			for( size_t r = 0; r != rank; ++r )
			{
				dv += m_dv[r][n]*a[r];
			}
			velosites[n] = summation_k( velosites[n], dv*dt, &m_correction_vel[n] );
		}

		//exit(0);

		data()->advise_time( dt );
	}
	else
	{
		step_v( vertites, m_dv.front() );
		m_starter->step( dt );

		nbvertex_t* dxfront = m_dx.front();

		//#pragma omp parallel for
		for( size_t n = 0; n < count; ++n )
		{
			dxfront[n] = velosites[n];
		}
	}

	nbvertex_t* tx = m_dx.back();
	nbvertex_t* tv = m_dv.back();
	m_dx.erase( m_dx.begin() + rank - 1 );
	m_dv.erase( m_dv.begin() + rank - 1 );
	m_dx.insert( m_dx.begin(), tx );
	m_dv.insert( m_dv.begin(), tv );

//	qDebug() << m_dx[0] << m_dx[1] << m_dx[2] << m_dx[3] << m_dx[4];
}
