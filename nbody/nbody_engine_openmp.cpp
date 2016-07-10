#include "nbody_engine_openmp.h"
#include <QDebug>
#include <omp.h>

nbody_engine_openmp::nbody_engine_openmp()
{
	qDebug() << "OpenMP max threads" << omp_get_max_threads();
}

nbody_engine_openmp::~nbody_engine_openmp()
{
}

const char* nbody_engine_openmp::type_name() const
{
	return "nbody_engine_openmp";
}

void nbody_engine_openmp::fcompute( const nbcoord_t& t, const memory* _y, memory* _f, size_t yoff, size_t foff )
{
	Q_UNUSED(t);
	const smemory*	y = dynamic_cast<const  smemory*>( _y );
	smemory*		f = dynamic_cast<smemory*>( _f );

	if( y == NULL )
	{
		qDebug() << "y not is smemory";
		return;
	}
	if( f == NULL )
	{
		qDebug() << "f not is smemory";
		return;
	}

	advise_compute_count();

	size_t				count = m_data->get_count();
	const nbcoord_t*	rx = ((const nbcoord_t*)y->data()) + yoff;
	const nbcoord_t*	ry = rx + count;
	const nbcoord_t*	rz = rx + 2*count;
	const nbcoord_t*	vx = rx + 3*count;
	const nbcoord_t*	vy = rx + 4*count;
	const nbcoord_t*	vz = rx + 5*count;

	nbcoord_t*			frx = ((nbcoord_t*)f->data()) + foff;
	nbcoord_t*			fry = frx + count;
	nbcoord_t*			frz = frx + 2*count;
	nbcoord_t*			fvx = frx + 3*count;
	nbcoord_t*			fvy = frx + 4*count;
	nbcoord_t*			fvz = frx + 5*count;

	const nbcoord_t*	mass = (const nbcoord_t*)m_mass->data();

	#pragma omp parallel for
	for( size_t body1 = 0; body1 < count; ++body1 )
	{
		const nbvertex_t	v1( rx[ body1 ], ry[ body1 ], rz[ body1 ] );
		nbvertex_t			total_force;
		for( size_t body2 = 0; body2 != count; ++body2 )
		{
			if( body1 == body2 )
				continue;
			const nbvertex_t	v2( rx[ body2 ], ry[ body2 ], rz[ body2 ] );
			const nbvertex_t	f( m_data->force( v1, v2, mass[body1], mass[body2] ) );
			total_force += f;
		}
		frx[body1] = vx[body1];
		fry[body1] = vy[body1];
		frz[body1] = vz[body1];
		fvx[body1] = total_force.x/mass[body1];
		fvy[body1] = total_force.y/mass[body1];
		fvz[body1] = total_force.z/mass[body1];
	}
}

void nbody_engine_openmp::memcpy( nbody_engine::memory* __a, const nbody_engine::memory* __b, size_t aoff, size_t boff )
{
	smemory*			_a = dynamic_cast<smemory*>( __a );
	const smemory*		_b = dynamic_cast<const smemory*>( __b );
	nbcoord_t*			a = (nbcoord_t*)_a->data();
	const nbcoord_t*	b = (const nbcoord_t*)_b->data();
	size_t				count = problem_size();

	#pragma omp parallel for
	for( size_t i = 0; i < count; ++i )
	{
		a[i + aoff] = b[i + boff];
	}
}

void nbody_engine_openmp::fmadd( nbody_engine::memory* __a, const nbody_engine::memory* __b, const nbcoord_t& c )
{
	smemory*			_a = dynamic_cast<smemory*>( __a );
	const smemory*		_b = dynamic_cast<const smemory*>( __b );
	nbcoord_t*			a = (nbcoord_t*)_a->data();
	const nbcoord_t*	b = (const nbcoord_t*)_b->data();
	size_t				count = problem_size();

	#pragma omp parallel for
	for( size_t i = 0; i < count; ++i )
	{
		a[i] += b[i]*c;
	}
}

void nbody_engine_openmp::fmadd( nbody_engine::memory* __a, const nbody_engine::memory* __b, const nbody_engine::memory* __c, const nbcoord_t& d, size_t aoff, size_t boff, size_t coff )
{
	smemory*			_a = dynamic_cast<smemory*>( __a );
	const smemory*		_b = dynamic_cast<const smemory*>( __b );
	const smemory*		_c = dynamic_cast<const smemory*>( __c );
	nbcoord_t*			a = (nbcoord_t*)_a->data();
	const nbcoord_t*	b = (const nbcoord_t*)_b->data();
	const nbcoord_t*	c = (const nbcoord_t*)_c->data();
	size_t				count = problem_size();

	#pragma omp parallel for
	for( size_t i = 0; i < count; ++i )
	{
		a[i + aoff] = b[i + boff] + c[i + coff]*d;
	}
}

void nbody_engine_openmp::fmaddn(nbody_engine::memory* __a, const nbody_engine::memory* __b, const nbody_engine::memory* __c, size_t bstride, size_t aoff, size_t boff, size_t csize)
{
	smemory*			_a = dynamic_cast<smemory*>( __a );
	const smemory*		_b = dynamic_cast<const smemory*>( __b );
	const smemory*		_c = dynamic_cast<const smemory*>( __c );
	nbcoord_t*			a = (nbcoord_t*)_a->data();
	const nbcoord_t*	b = (const nbcoord_t*)_b->data();
	const nbcoord_t*	c = (const nbcoord_t*)_c->data();
	size_t				count = problem_size();

	#pragma omp parallel for
	for( size_t i = 0; i < count; ++i )
	{
		nbcoord_t	sum = b[i+boff]*c[0];
		for( size_t k = 1; k < csize; ++k )
		{
			sum += b[ i + boff + k*bstride ]*c[k];
		}
		a[i+aoff] += sum;
	}
}

void nbody_engine_openmp::fmaddn( nbody_engine::memory* __a, const nbody_engine::memory* __b, const nbody_engine::memory* __c, const nbody_engine::memory* __d, size_t cstride, size_t aoff, size_t boff, size_t coff, size_t dsize )
{
	if( __b != NULL )
	{
		smemory*			_a = dynamic_cast<smemory*>( __a );
		const smemory*		_b = dynamic_cast<const smemory*>( __b );
		const smemory*		_c = dynamic_cast<const smemory*>( __c );
		const smemory*		_d = dynamic_cast<const smemory*>( __d );
		nbcoord_t*			a = (nbcoord_t*)_a->data();
		const nbcoord_t*	b = (const nbcoord_t*)_b->data();
		const nbcoord_t*	c = (const nbcoord_t*)_c->data();
		const nbcoord_t*	d = (const nbcoord_t*)_d->data();
		size_t				count = problem_size();

		#pragma omp parallel for
		for( size_t i = 0; i < count; ++i )
		{
			nbcoord_t	sum = c[i + coff]*d[0];
			for( size_t k = 1; k < dsize; ++k )
			{
				sum += c[ i + coff + k*cstride ]*d[k];
			}
			a[i+aoff] = b[i + boff] + sum;
		}
	}
	else
	{
		smemory*			_a = dynamic_cast<smemory*>( __a );
		const smemory*		_c = dynamic_cast<const smemory*>( __c );
		const smemory*		_d = dynamic_cast<const smemory*>( __d );
		nbcoord_t*			a = (nbcoord_t*)_a->data();
		const nbcoord_t*	c = (const nbcoord_t*)_c->data();
		const nbcoord_t*	d = (const nbcoord_t*)_d->data();
		size_t				count = problem_size();

		#pragma omp parallel for
		for( size_t i = 0; i < count; ++i )
		{
			nbcoord_t	sum = c[i + coff]*d[0];
			for( size_t k = 1; k < dsize; ++k )
			{
				sum += c[ i + coff + k*cstride ]*d[k];
			}
			a[i+aoff] = sum;
		}
	}
}

void nbody_engine_openmp::fmaxabs( const nbody_engine::memory* __a, nbcoord_t& result )
{
	const smemory*		_a = dynamic_cast<const smemory*>( __a );
	const nbcoord_t*	a = (nbcoord_t*)_a->data();
	size_t				count = problem_size();

	result = fabs(a[0]);

#if __GNUC__*100 + __GNUC_MINOR__ >= 409
	#pragma omp parallel for reduction( max : result )
#endif // since gcc-4.9
	for( size_t n = 0; n < count; ++n )
	{
		nbcoord_t	v( fabs( a[n] ) );
		if( v > result )
		{
			result = v;
		}
	}
}
