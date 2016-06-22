#include "nbody_engine_block.h"
#include <omp.h>
#include <QDebug>

nbody_engine_block::nbody_engine_block()
{
	qDebug() << "OpenMP max threads" << omp_get_max_threads();
}

void nbody_engine_block::fcompute( const nbody_data* data, const nbvertex_t* vertites, nbvertex_t* dv )
{
	advise_compute_count();

	size_t				count = data->get_count();
	const nbcoord_t*	mass = data->get_mass();
	const size_t		block = NBODY_DATA_BLOCK_SIZE;

	#pragma omp parallel for
	for( size_t n1 = 0; n1 < count; n1 += block )
	{
		const nbvertex_t*	v1( vertites + n1 );
		nbcoord_t			x1[block];
		nbcoord_t			y1[block];
		nbcoord_t			z1[block];
		nbcoord_t			total_force_x[block];
		nbcoord_t			total_force_y[block];
		nbcoord_t			total_force_z[block];

		for( size_t b1 = 0; b1 != block; ++b1 )
		{
			x1[b1] = v1[b1].x;
			y1[b1] = v1[b1].y;
			z1[b1] = v1[b1].z;
			total_force_x[b1] = 0;
			total_force_y[b1] = 0;
			total_force_z[b1] = 0;
		}
		for( size_t n2 = 0; n2 < count; n2 += block )
		{
			const nbvertex_t*	v2( vertites + n2 );
			nbcoord_t			x2[block];
			nbcoord_t			y2[block];
			nbcoord_t			z2[block];
			nbcoord_t			m2[block];

			for( size_t b2 = 0; b2 != block; ++b2 )
			{
				x2[b2] = v2[b2].x;
				y2[b2] = v2[b2].y;
				z2[b2] = v2[b2].z;
				m2[b2] = mass[n2 + b2];
			}

			for( size_t b1 = 0; b1 != block; ++b1 )
			{
				for( size_t b2 = 0; b2 != block; ++b2 )
				{
					nbcoord_t		dx = x1[b1] - x2[b2];
					nbcoord_t		dy = y1[b1] - y2[b2];
					nbcoord_t		dz = z1[b1] - z2[b2];
					nbcoord_t		r2( dx*dx + dy*dy + dz*dz );
					if( r2 < NBODY_MIN_R )
						r2 = NBODY_MIN_R;
					nbcoord_t		r = sqrt( r2 );
					nbcoord_t		coeff = (m2[b2])/(r*r2);

					dx *= coeff;
					dy *= coeff;
					dz *= coeff;

					total_force_x[b1] -= dx;
					total_force_y[b1] -= dy;
					total_force_z[b1] -= dz;
				}
			}
		}

		for( size_t b1 = 0; b1 != block; ++b1 )
		{
			dv[n1 + b1] = nbvertex_t( total_force_x[b1], total_force_y[b1], total_force_z[b1] );
		}
	}
}
