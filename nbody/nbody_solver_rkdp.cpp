#include "nbody_solver_rkdp.h"
#include <QDebug>

nbody_solver_rkdp::nbody_solver_rkdp( nbody_data* data ) :
	nbody_solver( data )
{
	m_step_subdivisions = 8;
	m_vrt_error_threshold = 1e-4;
	m_vel_error_threshold = 1e-4;
}

void nbody_solver_rkdp::step( double dt )
{
	nbvertex_t*	vertites = data()->get_vertites();
	nbvertex_t*	velosites = data()->get_velosites();

	sub_step( 1, dt, vertites, velosites, 0 );

	data()->advise_time( dt );
}

void nbody_solver_rkdp::sub_step( size_t substeps_count, nbcoord_t dt, nbvertex_t* vertites, nbvertex_t* velosites, size_t recursion_level )
{
#if 1
	static const size_t		STEPS = 7;//Dormand–Prince method
	static const nbcoord_t	c[]  = { 0, 1.0/5.0, 3.0/10.0, 4.0/5.0, 8.0/9.0, 1.0, 1.0 };
	static const nbcoord_t	b1[] = { 35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0 };
	static const nbcoord_t	b2[] = { 5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0 };
	static const nbcoord_t	a1[] = { 0 };
	static const nbcoord_t	a2[] = { 1.0/5.0 };
	static const nbcoord_t	a3[] = { 3.0/40.0, 9.0/40.0};
	static const nbcoord_t	a4[] = { 44.0/45.0, -56.0/15.0, 32.0/9.0 };
	static const nbcoord_t	a5[] = { 19372.0/6561.0, -25360.0/2187.0, 64448.0/6561.0, -212.0/729.0 };
	static const nbcoord_t	a6[] = { 9017.0/3168.0, -355.0/33.0, 46732.0/5247.0, 49.0/176.0, -5103.0/18656.0 };
	static const nbcoord_t	a7[] = { 35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0 };
	static const nbcoord_t*	a[] = { a1, a2, a3, a4, a5, a6, a7 };
#else
	static const size_t		STEPS = 4;//Classic RK4 Butcher table
	static const nbcoord_t	c[]  = { 0, 0.5, 0.5, 1.0 };
	static const nbcoord_t	b1[] = { 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0 };
	static const nbcoord_t	b2[] = { 1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0 };
	static const nbcoord_t	a1[] = { 0.0, 0.0 };
	static const nbcoord_t	a2[] = { 1.0/2.0, 0.0 };
	static const nbcoord_t	a3[] = { 0.0, 1.0/2.0, 0.0 };
	static const nbcoord_t	a4[] = { 0.0, 0.0, 1.0, 0.0 };
	static const nbcoord_t*	a[] = { a1, a2, a3, a4 };
#endif
	size_t		count = data()->get_count();


	if( k1.size() != count )
	{
		k1.resize( count );
		k2.resize( count );
		k3.resize( count );
		k4.resize( count );
		k5.resize( count );
		k6.resize( count );
		k7.resize( count );
		q1.resize( count );
		q2.resize( count );
		q3.resize( count );
		q4.resize( count );
		q5.resize( count );
		q6.resize( count );
		q7.resize( count );

		tmpvrt.resize( count );
		tmpvel.resize( count );
		for( size_t r = 0; r != MAX_RECURSION; ++r )
		{
			vertites_stack[r].resize( count );
			velosites_stack[r].resize( count );
		}
	}

	nbvertex_t*	k[] = { k1.data(), k2.data(), k3.data(), k4.data(), k5.data(), k6.data(), k7.data() };
	nbvertex_t*	q[] = { q1.data(), q2.data(), q3.data(), q4.data(), q5.data(), q6.data(), q7.data() };

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

		bool can_subdivide = ( recursion_level < MAX_RECURSION ) && dt > get_min_step();
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
