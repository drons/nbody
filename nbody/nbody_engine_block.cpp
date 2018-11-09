#include "nbody_engine_block.h"
#include <omp.h>
#include <QDebug>

nbody_engine_block::nbody_engine_block()
{
}

const char* nbody_engine_block::type_name() const
{
	return "nbody_engine_block";
}

void nbody_engine_block::fcompute(const nbcoord_t& t, const memory* _y, memory* _f)
{
	Q_UNUSED(t);
	const smemory*	y = dynamic_cast<const  smemory*>(_y);
	smemory*		f = dynamic_cast<smemory*>(_f);

	if(y == NULL)
	{
		qDebug() << "y is not smemory";
		return;
	}
	if(f == NULL)
	{
		qDebug() << "f is not smemory";
		return;
	}

	advise_compute_count();

	size_t				count = m_data->get_count();
	const size_t		block = NBODY_DATA_BLOCK_SIZE;

	const nbcoord_t*	rx = reinterpret_cast<const nbcoord_t*>(y->data());
	const nbcoord_t*	ry = rx + count;
	const nbcoord_t*	rz = rx + 2 * count;
	const nbcoord_t*	vx = rx + 3 * count;
	const nbcoord_t*	vy = rx + 4 * count;
	const nbcoord_t*	vz = rx + 5 * count;

	nbcoord_t*			frx = reinterpret_cast<nbcoord_t*>(f->data());
	nbcoord_t*			fry = frx + count;
	nbcoord_t*			frz = frx + 2 * count;
	nbcoord_t*			fvx = frx + 3 * count;
	nbcoord_t*			fvy = frx + 4 * count;
	nbcoord_t*			fvz = frx + 5 * count;
	const nbcoord_t*	mass = reinterpret_cast<const nbcoord_t*>(m_mass->data());

	#pragma omp parallel for
	for(size_t n1 = 0; n1 < count; n1 += block)
	{
		nbcoord_t			x1[block];
		nbcoord_t			y1[block];
		nbcoord_t			z1[block];
		nbcoord_t			total_force_x[block];
		nbcoord_t			total_force_y[block];
		nbcoord_t			total_force_z[block];

		for(size_t b1 = 0; b1 != block; ++b1)
		{
			size_t local_n1 = b1 + n1;

			x1[b1] = rx[local_n1];
			y1[b1] = ry[local_n1];
			z1[b1] = rz[local_n1];
			total_force_x[b1] = 0;
			total_force_y[b1] = 0;
			total_force_z[b1] = 0;
		}
		for(size_t n2 = 0; n2 < count; n2 += block)
		{
			nbcoord_t			x2[block];
			nbcoord_t			y2[block];
			nbcoord_t			z2[block];
			nbcoord_t			m2[block];

			for(size_t b2 = 0; b2 != block; ++b2)
			{
				size_t local_n2 = b2 + n2;

				x2[b2] = rx[local_n2];
				y2[b2] = ry[local_n2];
				z2[b2] = rz[local_n2];
				m2[b2] = mass[n2 + b2];
			}

			for(size_t b1 = 0; b1 != block; ++b1)
			{
				for(size_t b2 = 0; b2 != block; ++b2)
				{
					nbcoord_t		dx = x1[b1] - x2[b2];
					nbcoord_t		dy = y1[b1] - y2[b2];
					nbcoord_t		dz = z1[b1] - z2[b2];
					nbcoord_t		r2(dx * dx + dy * dy + dz * dz);
					if(r2 < NBODY_MIN_R)
					{
						r2 = NBODY_MIN_R;
					}
					nbcoord_t		r = sqrt(r2);
					nbcoord_t		coeff = (m2[b2]) / (r * r2);

					dx *= coeff;
					dy *= coeff;
					dz *= coeff;

					total_force_x[b1] -= dx;
					total_force_y[b1] -= dy;
					total_force_z[b1] -= dz;
				}
			}
		}

		for(size_t b1 = 0; b1 != block; ++b1)
		{
			size_t local_n1 = b1 + n1;
			frx[local_n1] = vx[local_n1];
			fry[local_n1] = vy[local_n1];
			frz[local_n1] = vz[local_n1];
			fvx[local_n1] = total_force_x[b1];
			fvy[local_n1] = total_force_y[b1];
			fvz[local_n1] = total_force_z[b1];
		}
	}
}
