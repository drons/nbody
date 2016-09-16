#include "nbody_engine_simple.h"
#include <QDebug>

nbody_engine_simple::nbody_engine_simple()
{
	m_mass = NULL;
	m_y = NULL;
	m_data = NULL;
}

nbody_engine_simple::~nbody_engine_simple()
{
	free_buffer( m_mass );
	free_buffer( m_y );
}

const char* nbody_engine_simple::type_name() const
{
	return "nbody_engine_simple";
}

void nbody_engine_simple::init( nbody_data* data )
{
	m_data = data;
	m_mass = create_buffer( sizeof(nbcoord_t)*m_data->get_count() );
	m_y = create_buffer( sizeof( nbcoord_t )*problem_size() );

	size_t		count = m_data->get_count();
	nbcoord_t*	m = (nbcoord_t*)m_mass->data();
	nbcoord_t*	rx = (nbcoord_t*)m_y->data();
	nbcoord_t*	ry = rx + count;
	nbcoord_t*	rz = rx + 2*count;
	nbcoord_t*	vx = rx + 3*count;
	nbcoord_t*	vy = rx + 4*count;
	nbcoord_t*	vz = rx + 5*count;
	const nbvertex_t*	vrt = data->get_vertites();
	const nbvertex_t*	vel = data->get_velosites();
	const nbcoord_t	*	mass = data->get_mass();

	for( size_t i = 0; i != count; ++i )
	{
		rx[i] = vrt[i].x;
		ry[i] = vrt[i].y;
		rz[i] = vrt[i].z;
		vx[i] = vel[i].x;
		vy[i] = vel[i].y;
		vz[i] = vel[i].z;
		m[i] = mass[i];
	}
}

void nbody_engine_simple::get_data(nbody_data* data)
{
	size_t				count = m_data->get_count();
	const nbcoord_t*	rx = (const nbcoord_t*)m_y->data();
	const nbcoord_t*	ry = rx + count;
	const nbcoord_t*	rz = rx + 2*count;
	const nbcoord_t*	vx = rx + 3*count;
	const nbcoord_t*	vy = rx + 4*count;
	const nbcoord_t*	vz = rx + 5*count;
	nbvertex_t*			vrt = data->get_vertites();
	nbvertex_t*			vel = data->get_velosites();

	for( size_t i = 0; i != count; ++i )
	{
		vrt[i].x = rx[i];
		vrt[i].y = ry[i];
		vrt[i].z = rz[i];
		vel[i].x = vx[i];
		vel[i].y = vy[i];
		vel[i].z = vz[i];
	}
}

size_t nbody_engine_simple::problem_size() const
{
	return 6*m_data->get_count();
}

nbody_engine::memory*nbody_engine_simple::y()
{
	return m_y;
}

void nbody_engine_simple::advise_time( const nbcoord_t& dt )
{
	m_data->advise_time( dt );
}

nbcoord_t nbody_engine_simple::get_time() const
{
	return m_data->get_time();
}

void nbody_engine_simple::set_time( nbcoord_t t )
{
	m_data->set_time( t );
}

size_t nbody_engine_simple::get_step() const
{
	return m_data->get_step();
}

void nbody_engine_simple::set_step( size_t s )
{
	m_data->set_step( s );
}

void nbody_engine_simple::fcompute( const nbcoord_t& t, const memory* _y, memory* _f, size_t yoff, size_t foff )
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

nbody_engine_simple::smemory* nbody_engine_simple::create_buffer( size_t s )
{
	return new smemory( s );
}

void nbody_engine_simple::free_buffer( memory* m )
{
	delete m;
}

void nbody_engine_simple::read_buffer( void* dst, memory* _src )
{
	smemory*		src = dynamic_cast<smemory*>( _src );
	::memcpy( dst, src->data(), src->size() );
}

void nbody_engine_simple::write_buffer( memory* _dst, void* src )
{
	smemory*		dst = dynamic_cast<smemory*>( _dst );
	::memcpy( dst->data(), src, dst->size() );
}

void nbody_engine_simple::copy_buffer( nbody_engine::memory* __a, const nbody_engine::memory* __b, size_t aoff, size_t boff )
{
	smemory*			_a = dynamic_cast<smemory*>( __a );
	const smemory*		_b = dynamic_cast<const smemory*>( __b );
	nbcoord_t*			a = (nbcoord_t*)_a->data();
	const nbcoord_t*	b = (const nbcoord_t*)_b->data();
	size_t				count = problem_size();

	for( size_t i = 0; i < count; ++i )
	{
		a[i + aoff] = b[i + boff];
	}
}

void nbody_engine_simple::fmadd_inplace( memory* __a, const memory* __b, const nbcoord_t& c )
{
	smemory*			_a = dynamic_cast<smemory*>( __a );
	const smemory*		_b = dynamic_cast<const smemory*>( __b );
	nbcoord_t*			a = (nbcoord_t*)_a->data();
	const nbcoord_t*	b = (const nbcoord_t*)_b->data();
	size_t				count = problem_size();

	for( size_t i = 0; i < count; ++i )
	{
		a[i] += b[i]*c;
	}
}

void nbody_engine_simple::fmadd( memory* __a, const memory* __b, const memory* __c, const nbcoord_t& d, size_t aoff, size_t boff, size_t coff )
{
	smemory*			_a = dynamic_cast<smemory*>( __a );
	const smemory*		_b = dynamic_cast<const smemory*>( __b );
	const smemory*		_c = dynamic_cast<const smemory*>( __c );
	nbcoord_t*			a = (nbcoord_t*)_a->data();
	const nbcoord_t*	b = (const nbcoord_t*)_b->data();
	const nbcoord_t*	c = (const nbcoord_t*)_c->data();
	size_t				count = problem_size();

	for( size_t i = 0; i < count; ++i )
	{
		a[i + aoff] = b[i + boff] + c[i + coff]*d;
	}
}

void nbody_engine_simple::fmaddn_inplace(nbody_engine::memory* __a, const nbody_engine::memory* __b, const nbody_engine::memory* __c, size_t bstride, size_t aoff, size_t boff, size_t csize)
{
	smemory*			_a = dynamic_cast<smemory*>( __a );
	const smemory*		_b = dynamic_cast<const smemory*>( __b );
	const smemory*		_c = dynamic_cast<const smemory*>( __c );
	nbcoord_t*			a = (nbcoord_t*)_a->data();
	const nbcoord_t*	b = (const nbcoord_t*)_b->data();
	const nbcoord_t*	c = (const nbcoord_t*)_c->data();
	size_t				count = problem_size();

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

void nbody_engine_simple::fmaddn( nbody_engine::memory* __a, const nbody_engine::memory* __b, const nbody_engine::memory* __c, const nbody_engine::memory* __d, size_t cstride, size_t aoff, size_t boff, size_t coff, size_t dsize )
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

void nbody_engine_simple::fmaxabs( const nbody_engine::memory* __a, nbcoord_t& result )
{
	const smemory*		_a = dynamic_cast<const smemory*>( __a );
	const nbcoord_t*	a = (nbcoord_t*)_a->data();
	size_t				count = problem_size();

	result = fabs(a[0]);

	for( size_t n = 0; n < count; ++n )
	{
		nbcoord_t	v( fabs( a[n] ) );
		if( v > result )
		{
			result = v;
		}
	}
}

nbody_engine_simple::smemory::smemory( size_t s )
{
	m_data = ::malloc( s );
	m_size = s;
}

nbody_engine_simple::smemory::~smemory()
{
	::free( m_data );
}

void* nbody_engine_simple::smemory::data() const
{
	return m_data;
}

size_t nbody_engine_simple::smemory::size() const
{
	return m_size;
}
