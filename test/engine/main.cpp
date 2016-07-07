#include <QApplication>
#include <QDebug>
#include <limits>

#include "nbody_engine_block.h"
#include "nbody_engine_opencl.h"
#include "nbody_engine_simple.h"
#include "nbody_engine_sparse.h"

int test_mem( nbody_engine* e )
{
	nbody_engine::memory*	mem = e->malloc( 1024 );

	Q_ASSERT( mem != NULL );

	e->free( mem );
	return 0;
}

int test_memcpy( nbody_engine* e )
{
	const size_t			cnt = 8;
	const size_t			size = sizeof( nbcoord_t )*cnt;
	nbcoord_t				data[cnt] = { 0,1,2,3,4,5,6,7 };
	nbcoord_t				x[cnt] = {0};
	nbody_engine::memory*	mem = e->malloc( size );
	nbody_engine::memory*	submem = e->malloc( size/2 );

	e->memcpy( mem, data );
	e->memcpy( x, mem );

	Q_ASSERT( 0 == memcmp( data, x, size ) );

	e->free( mem );
	e->free( submem );

	return 0;
}

int test_fmadd1( nbody_engine* e )
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	const size_t			size = sizeof( nbcoord_t )*e->problem_size();
	std::vector<nbcoord_t>	a( e->problem_size() );
	std::vector<nbcoord_t>	a_res( e->problem_size() );
	std::vector<nbcoord_t>	b( e->problem_size() );
	nbcoord_t				c = 5;
	nbody_engine::memory*	mem_a = e->malloc( size );
	nbody_engine::memory*	mem_b = e->malloc( size );

	for( size_t n = 0; n != a.size(); ++n )
	{
		a[n] = rand() % 10000;
		b[n] = rand() % 10000;
	}

	e->memcpy( mem_a, a.data() );
	e->memcpy( mem_b, b.data() );

	//! a[i] += b[i]*c
	e->fmadd( mem_a, mem_b, c );

	e->memcpy( a_res.data(), mem_a );

	for( size_t i = 0; i != a.size(); ++i )
	{
		Q_ASSERT( fabs( (a[i] + c*b[i]) - a_res[i] ) < eps );
	}

	e->free( mem_a );
	e->free( mem_b );

	return 0;
}

int test_fmadd2( nbody_engine* e )
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	const size_t			size = e->problem_size();
	size_t					aoff = 33;
	size_t					boff = 44;
	size_t					coff = 55;
	std::vector<nbcoord_t>	a( e->problem_size() + aoff );
	std::vector<nbcoord_t>	b( e->problem_size() + boff );
	std::vector<nbcoord_t>	c( e->problem_size() + coff );
	nbcoord_t				d = 5;
	nbody_engine::memory*	mem_a = e->malloc( a.size()*sizeof( nbcoord_t ) );
	nbody_engine::memory*	mem_b = e->malloc( b.size()*sizeof( nbcoord_t ) );
	nbody_engine::memory*	mem_c = e->malloc( c.size()*sizeof( nbcoord_t ) );

	for( size_t n = 0; n != a.size(); ++n )
	{
		a[n] = rand() % 10000;
	}
	for( size_t n = 0; n != b.size(); ++n )
	{
		b[n] = rand() % 10000;
	}
	for( size_t n = 0; n != c.size(); ++n )
	{
		c[n] = rand() % 10000;
	}

	e->memcpy( mem_a, a.data() );
	e->memcpy( mem_b, b.data() );
	e->memcpy( mem_c, c.data() );

	//! a[i+aoff] = b[i+boff] + c[i+coff]*d
	e->fmadd( mem_a, mem_b, mem_c, d, aoff, boff, coff );

	e->memcpy( a.data(), mem_a );

	for( size_t i = 0; i != size; ++i )
	{
		Q_ASSERT( fabs( (b[i+boff] + c[i+coff]*d) - a[i+aoff] ) < eps );
	}

	e->free( mem_a );
	e->free( mem_b );
	e->free( mem_c );

	return 0;
}

int test_fmaddn1( nbody_engine* e, size_t csize )
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	const size_t			size = e->problem_size();
	const size_t			bstride = size;
	size_t					aoff = 33;
	size_t					boff = 44;
	std::vector<nbcoord_t>	a( e->problem_size() + aoff );
	std::vector<nbcoord_t>	a_res( e->problem_size() + aoff );
	std::vector<nbcoord_t>	b( e->problem_size()*csize + boff );
	std::vector<nbcoord_t>	c( csize );
	nbody_engine::memory*	mem_a = e->malloc( a.size()*sizeof( nbcoord_t ) );
	nbody_engine::memory*	mem_b = e->malloc( b.size()*sizeof( nbcoord_t ) );
	nbody_engine::memory*	mem_c = e->malloc( c.size()*sizeof( nbcoord_t ) );

	for( size_t n = 0; n != a.size(); ++n )
	{
		a[n] = rand() % 10000;
	}
	for( size_t n = 0; n != b.size(); ++n )
	{
		b[n] = rand() % 10000;
	}
	for( size_t n = 0; n != c.size(); ++n )
	{
		c[n] = rand() % 10000;
	}

	e->memcpy( mem_a, a.data() );
	e->memcpy( mem_b, b.data() );
	e->memcpy( mem_c, c.data() );

	//! a[i+aoff] += sum( b[i+boff+k*bstride]*c[k], k=[0...csize) )
	e->fmaddn( mem_a, mem_b, mem_c, bstride, aoff, boff, csize );

	e->memcpy( a_res.data(), mem_a );

	for( size_t i = 0; i != size; ++i )
	{
		nbcoord_t	s = 0;
		for( size_t k = 0; k != csize; ++k )
			s += b[i+boff+k*bstride]*c[k];
		Q_ASSERT( fabs( (a[i+aoff] + s) - a_res[i+aoff] ) < eps );
	}

	e->free( mem_a );
	e->free( mem_b );
	e->free( mem_c );

	return 0;
}

int test_fmaddn2( nbody_engine* e, size_t dsize )
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	const size_t			size = e->problem_size();
	const size_t			cstride = size;
	size_t					aoff = 33;
	size_t					boff = 44;
	size_t					coff = 55;
	std::vector<nbcoord_t>	a( e->problem_size() + aoff );
	std::vector<nbcoord_t>	b( e->problem_size() + boff );
	std::vector<nbcoord_t>	c( e->problem_size()*dsize + coff );
	std::vector<nbcoord_t>	d( dsize );
	nbody_engine::memory*	mem_a = e->malloc( a.size()*sizeof( nbcoord_t ) );
	nbody_engine::memory*	mem_b = e->malloc( b.size()*sizeof( nbcoord_t ) );
	nbody_engine::memory*	mem_c = e->malloc( c.size()*sizeof( nbcoord_t ) );
	nbody_engine::memory*	mem_d = e->malloc( d.size()*sizeof( nbcoord_t ) );

	for( size_t n = 0; n != a.size(); ++n )
	{
		a[n] = rand() % 10000;
	}
	for( size_t n = 0; n != b.size(); ++n )
	{
		b[n] = rand() % 10000;
	}
	for( size_t n = 0; n != c.size(); ++n )
	{
		c[n] = rand() % 10000;
	}
	for( size_t n = 0; n != d.size(); ++n )
	{
		d[n] = rand() % 10000;
	}

	e->memcpy( mem_a, a.data() );
	e->memcpy( mem_b, b.data() );
	e->memcpy( mem_c, c.data() );
	e->memcpy( mem_d, d.data() );

	//! a[i+aoff] = b[i+boff] + sum( c[i+coff+k*cstride]*d[k], k=[0...dsize) )
	e->fmaddn( mem_a, mem_b, mem_c, mem_d, cstride, aoff, boff, coff, dsize );

	e->memcpy( a.data(), mem_a );

	for( size_t i = 0; i != size; ++i )
	{
		nbcoord_t	s = 0;
		for( size_t k = 0; k != dsize; ++k )
			s += c[i+coff+k*cstride]*d[k];
		Q_ASSERT( fabs( (b[i+boff] + s) - a[i+aoff] ) < eps );
	}

	e->free( mem_a );
	e->free( mem_b );
	e->free( mem_c );
	e->free( mem_d );

	return 0;
}

int test_fmaddn3( nbody_engine* e, size_t dsize )
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	const size_t			size = e->problem_size();
	const size_t			cstride = size;
	size_t					aoff = 0;
	size_t					coff = 0;
	std::vector<nbcoord_t>	a( e->problem_size() + aoff );
	std::vector<nbcoord_t>	c( e->problem_size()*dsize + coff );
	std::vector<nbcoord_t>	d( dsize );
	nbody_engine::memory*	mem_a = e->malloc( a.size()*sizeof( nbcoord_t ) );
	nbody_engine::memory*	mem_c = e->malloc( c.size()*sizeof( nbcoord_t ) );
	nbody_engine::memory*	mem_d = e->malloc( d.size()*sizeof( nbcoord_t ) );

	for( size_t n = 0; n != a.size(); ++n )
	{
		a[n] = rand() % 10000;
	}
	for( size_t n = 0; n != c.size(); ++n )
	{
		c[n] = rand() % 10000;
	}
	for( size_t n = 0; n != d.size(); ++n )
	{
		d[n] = rand() % 10000;
	}

	e->memcpy( mem_a, a.data() );
	e->memcpy( mem_c, c.data() );
	e->memcpy( mem_d, d.data() );

	//! a[i+aoff] = b[i+boff] + sum( c[i+coff+k*cstride]*d[k], k=[0...dsize) )
	e->fmaddn( mem_a, NULL, mem_c, mem_d, cstride, aoff, 0, coff, dsize );

	e->memcpy( a.data(), mem_a );

	for( size_t i = 0; i != size; ++i )
	{
		nbcoord_t	s = 0;
		for( size_t k = 0; k < dsize; ++k )
		{
			s += c[i+coff+k*cstride]*d[k];
		}
		Q_ASSERT( fabs( s - a[i+aoff] ) < eps );
	}

	e->free( mem_a );
	e->free( mem_c );
	e->free( mem_d );

	return 0;
}

int test_fmaxabs( nbody_engine* e )
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	std::vector<nbcoord_t>	a( e->problem_size() );
	nbody_engine::memory*	mem_a = e->malloc( sizeof(nbcoord_t)*a.size() );
	nbcoord_t				result = 2878767678687;//Garbage

	for( size_t n = 0; n != a.size(); ++n )
	{
		a[n] = rand() % 10000;
	}

	e->memcpy( mem_a, a.data() );

	//! @result = max( fabs(a[k]), k=[0...asize) )
	e->fmaxabs( mem_a, result );

	std::transform( a.begin(), a.end(), a.begin(), fabs );
	nbcoord_t testmax = *std::max_element( a.begin(), a.end() );

	Q_ASSERT( fabs( result - testmax ) < eps );

	e->free( mem_a );

	return 0;
}

int test_engine( nbody_engine* e )
{
	nbody_data              data;
	nbcoord_t				box_size = 100;

	data.make_universe( box_size, box_size, box_size );

	e->init( &data );

	test_mem( e );
	test_memcpy( e );

	test_fmadd1( e );
	test_fmadd2( e );

	test_fmaddn1( e, 1 );
	test_fmaddn1( e, 3 );
	test_fmaddn1( e, 7 );

	test_fmaddn2( e, 1 );
	test_fmaddn2( e, 3 );
	test_fmaddn2( e, 7 );

	test_fmaddn3( e, 1 );
	test_fmaddn3( e, 3 );
	test_fmaddn3( e, 7 );

	test_fmaxabs( e );

	return 0;
}

int main( int argc, char *argv[] )
{
//	if( argc != 2 )
//	{
//		qDebug() << "invalid arg count";
//		qDebug() << "Usage: " << argv[0] << "engine_name";
//	}

	{
		nbody_engine_block	e;
		test_engine( &e );
	}
	{
		nbody_engine_opencl	e;
		test_engine( &e );
	}
	{
		nbody_engine_simple	e;
		test_engine( &e );
	}
//	{
//		nbody_engine_sparse	e;
//		test_engine( &e );
//	}

	return 0;
}
