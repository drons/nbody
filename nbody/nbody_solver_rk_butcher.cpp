#include "nbody_solver_rk_butcher.h"
#include <QDebug>

nbody_solver_rk_butcher::nbody_solver_rk_butcher( nbody_data* data, nbody_butcher_table* t ) :
	nbody_solver( data )
{
	m_bt = t;
	m_step_subdivisions = 8;
	m_vrt_error_threshold = 1e-4;
	m_vel_error_threshold = 1e-4;
}

nbody_solver_rk_butcher::~nbody_solver_rk_butcher()
{
	delete m_bt;
}

void nbody_solver_rk_butcher::step( double dt )
{
	nbvertex_t*	vertites = data()->get_vertites();
	nbvertex_t*	velosites = data()->get_velosites();

	sub_step( 1, dt, vertites, velosites, 0 );

	data()->advise_time( dt );
}

void nbody_solver_rk_butcher::sub_step( size_t substeps_count, nbcoord_t dt, nbvertex_t* vertites, nbvertex_t* velosites, size_t recursion_level )
{
	const size_t		STEPS = m_bt->get_steps();
	const nbcoord_t**	a = m_bt->get_a();
	const nbcoord_t*	b1 = m_bt->get_b1();
	const nbcoord_t*	b2 = m_bt->get_b2();
	const nbcoord_t*	c = m_bt->get_c();
	size_t		count = data()->get_count();

	if( tmpvel.size() != count )
	{
		m_k.resize( m_bt->get_steps() );
		m_q.resize( m_bt->get_steps() );
		m_kptr.resize( m_bt->get_steps() );
		m_qptr.resize( m_bt->get_steps() );

		for( size_t s = 0; s != m_bt->get_steps(); ++s )
		{
			m_k[s].resize( count );
			m_q[s].resize( count );
			m_kptr[s] = m_k[s].data();
			m_qptr[s] = m_q[s].data();
		}

		tmpvrt.resize( count );
		tmpvel.resize( count );
		for( size_t r = 0; r != MAX_RECURSION; ++r )
		{
			vertites_stack[r].resize( count );
			velosites_stack[r].resize( count );
		}
	}


	nbvertex_t**	k = m_kptr.data();
	nbvertex_t**	q = m_qptr.data();

	for( size_t sub_n = 0; sub_n != substeps_count; ++sub_n )
	{
		for( size_t i = 0; i < STEPS; ++i )
		{
			if( i == 0 )
			{
				step_v( vertites, k[i] );

				#pragma omp parallel for
				for( size_t n = 0; n < count; ++n )
				{
					q[i][n] = velosites[n];
				}
			}
			else
			{
				#pragma omp parallel for
				for( size_t n = 0; n < count; ++n )
				{
					nbvertex_t	qsum;
					nbvertex_t	ksum;
					for( size_t j = 0; j != i; ++j )
					{
						qsum += q[j][n]*a[i][j];
						ksum += k[j][n]*a[i][j];
					}
					tmpvrt[n] = vertites[n] + qsum*dt;
					tmpvel[n] = velosites[n] + ksum*dt;
				}

				step_v( tmpvrt.data(), k[i] );
				#pragma omp parallel for
				for( size_t n = 0; n < count; ++n )
				{
					q[i][n] = tmpvel[n];
				}
			}
		}

		#pragma omp parallel for
		for( size_t n = 0; n < count; ++n )
		{
			nbvertex_t	dvel1;
			nbvertex_t	dvrt1;
			nbvertex_t	dvel2;
			nbvertex_t	dvrt2;
			for( size_t i = 0; i < STEPS; ++i )
			{
				dvel1 += k[i][n]*b1[i];
				dvrt1 += q[i][n]*b1[i];
				dvel2 += k[i][n]*b2[i];
				dvrt2 += q[i][n]*b2[i];
			}

			tmpvel[n] = dvel2 - dvel1;
			tmpvrt[n] = dvrt2 - dvrt1;
		}

		nbcoord_t dvrt_max = 0.0;
		nbcoord_t dvel_max = 0.0;

		if( m_bt->is_embedded() )
		{
			#pragma omp parallel for reduction( max : dvrt_max )
			for( size_t n = 0; n < count; ++n )
			{
				nbcoord_t	local_max = tmpvrt[n].length();
				if( local_max > dvrt_max )
				{
					dvrt_max = local_max;
				}
			}
			#pragma omp parallel for reduction( max : dvel_max )
			for( size_t n = 0; n < count; ++n )
			{
				nbcoord_t	local_max = tmpvel[n].length();
				if( local_max > dvel_max )
				{
					dvel_max = local_max;
				}
			}
		}

		bool can_subdivide = ( m_bt->is_embedded() && recursion_level < MAX_RECURSION ) && dt > get_min_step();
		bool need_subdivide = dvel_max > m_vel_error_threshold || dvrt_max > m_vrt_error_threshold;

		if( can_subdivide && need_subdivide )
		{
			nbcoord_t	new_dt = dt/m_step_subdivisions;

			qDebug() << data()->get_step() << QString( "-" ).repeated(recursion_level) << "sub_step #" << sub_n << "ERR" << dvrt_max << dvel_max << "Down to dt" << new_dt;

			nbvertex_t*	vrt_head = vertites_stack[recursion_level].data();
			nbvertex_t*	vel_head = velosites_stack[recursion_level].data();

			#pragma omp parallel for
			for( size_t n = 0; n < count; ++n )
			{
				vel_head[n] = velosites[n];
				vrt_head[n] = vertites[n];
			}

			sub_step( m_step_subdivisions, new_dt, vrt_head, vel_head, recursion_level + 1 );
			#pragma omp parallel for
			for( size_t n = 0; n < count; ++n )
			{
				velosites[n] = vel_head[n];
				vertites[n] = vrt_head[n];
			}
		}
		else
		{
			#pragma omp parallel for
			for( size_t n = 0; n < count; ++n )
			{
				nbvertex_t	dvel2;
				nbvertex_t	dvrt2;
				for( size_t i = 0; i < STEPS; ++i )
				{
					dvel2 += k[i][n]*b2[i];
					dvrt2 += q[i][n]*b2[i];
				}

				velosites[n] += dvel2*dt;
				vertites[n] += dvrt2*dt;
			}
		}
	}//for( size_t sub_n = 0; sub_n != substeps_count; ++sub_n )
}
