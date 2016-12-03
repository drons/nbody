#include <QApplication>
#include <QtTest>
#include <QDebug>
#include <limits>

#include "nbody_engines.h"

bool test_mem( nbody_engine* e )
{
	nbody_engine::memory*	mem = e->create_buffer( 1024 );

	if( mem == NULL )
	{
		return false;
	}
	e->free_buffer( mem );

	return true;
}

bool test_memcpy( nbody_engine* e )
{
	const size_t			cnt = 8;
	const size_t			size = sizeof( nbcoord_t )*cnt;
	nbcoord_t				data[cnt] = { 0,1,2,3,4,5,6,7 };
	nbcoord_t				x[cnt] = {0};
	nbody_engine::memory*	mem = e->create_buffer( size );
	nbody_engine::memory*	submem = e->create_buffer( size/2 );

	e->write_buffer( mem, data );
	e->read_buffer( x, mem );

	bool	ret = ( 0 == memcmp( data, x, size ) );

	e->free_buffer( mem );
	e->free_buffer( submem );

	return ret;
}

bool test_fmadd1( nbody_engine* e )
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	const size_t			size = sizeof( nbcoord_t )*e->problem_size();
	std::vector<nbcoord_t>	a( e->problem_size() );
	std::vector<nbcoord_t>	a_res( e->problem_size() );
	std::vector<nbcoord_t>	b( e->problem_size() );
	nbcoord_t				c = 5;
	nbody_engine::memory*	mem_a = e->create_buffer( size );
	nbody_engine::memory*	mem_b = e->create_buffer( size );

	for( size_t n = 0; n != a.size(); ++n )
	{
		a[n] = rand() % 10000;
		b[n] = rand() % 10000;
	}

	e->write_buffer( mem_a, a.data() );
	e->write_buffer( mem_b, b.data() );

	//! a[i] += b[i]*c
	e->fmadd_inplace( mem_a, mem_b, c );

	e->read_buffer( a_res.data(), mem_a );

	bool	ret = true;
	for( size_t i = 0; i != a.size(); ++i )
	{
		if( fabs( (a[i] + c*b[i]) - a_res[i] ) > eps )
		{
			ret = false;
		}
	}

	e->free_buffer( mem_a );
	e->free_buffer( mem_b );

	return ret;
}

bool test_fmadd2( nbody_engine* e )
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
	nbody_engine::memory*	mem_a = e->create_buffer( a.size()*sizeof( nbcoord_t ) );
	nbody_engine::memory*	mem_b = e->create_buffer( b.size()*sizeof( nbcoord_t ) );
	nbody_engine::memory*	mem_c = e->create_buffer( c.size()*sizeof( nbcoord_t ) );

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

	e->write_buffer( mem_a, a.data() );
	e->write_buffer( mem_b, b.data() );
	e->write_buffer( mem_c, c.data() );

	//! a[i+aoff] = b[i+boff] + c[i+coff]*d
	e->fmadd( mem_a, mem_b, mem_c, d, aoff, boff, coff );

	e->read_buffer( a.data(), mem_a );

	bool	ret = true;
	for( size_t i = 0; i != size; ++i )
	{
		if( fabs( (b[i+boff] + c[i+coff]*d) - a[i+aoff] ) > eps )
		{
			ret = false;
		}
	}

	e->free_buffer( mem_a );
	e->free_buffer( mem_b );
	e->free_buffer( mem_c );

	return ret;
}

bool test_fmaddn1( nbody_engine* e, size_t csize )
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
	nbody_engine::memory*	mem_a = e->create_buffer( a.size()*sizeof( nbcoord_t ) );
	nbody_engine::memory*	mem_b = e->create_buffer( b.size()*sizeof( nbcoord_t ) );
	nbody_engine::memory*	mem_c = e->create_buffer( c.size()*sizeof( nbcoord_t ) );

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

	e->write_buffer( mem_a, a.data() );
	e->write_buffer( mem_b, b.data() );
	e->write_buffer( mem_c, c.data() );

	//! a[i+aoff] += sum( b[i+boff+k*bstride]*c[k], k=[0...csize) )
	e->fmaddn_inplace( mem_a, mem_b, mem_c, bstride, aoff, boff, csize );

	e->read_buffer( a_res.data(), mem_a );

	bool	ret = true;
	for( size_t i = 0; i != size; ++i )
	{
		nbcoord_t	s = 0;
		for( size_t k = 0; k != csize; ++k )
			s += b[i+boff+k*bstride]*c[k];
		if( fabs( (a[i+aoff] + s) - a_res[i+aoff] ) > eps )
		{
			ret = false;
		}
	}

	e->free_buffer( mem_a );
	e->free_buffer( mem_b );
	e->free_buffer( mem_c );

	return ret;
}

bool test_fmaddn2( nbody_engine* e, size_t dsize )
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
	nbody_engine::memory*	mem_a = e->create_buffer( a.size()*sizeof( nbcoord_t ) );
	nbody_engine::memory*	mem_b = e->create_buffer( b.size()*sizeof( nbcoord_t ) );
	nbody_engine::memory*	mem_c = e->create_buffer( c.size()*sizeof( nbcoord_t ) );
	nbody_engine::memory*	mem_d = e->create_buffer( d.size()*sizeof( nbcoord_t ) );

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

	e->write_buffer( mem_a, a.data() );
	e->write_buffer( mem_b, b.data() );
	e->write_buffer( mem_c, c.data() );
	e->write_buffer( mem_d, d.data() );

	//! a[i+aoff] = b[i+boff] + sum( c[i+coff+k*cstride]*d[k], k=[0...dsize) )
	e->fmaddn( mem_a, mem_b, mem_c, mem_d, cstride, aoff, boff, coff, dsize );

	e->read_buffer( a.data(), mem_a );

	bool	ret = true;
	for( size_t i = 0; i != size; ++i )
	{
		nbcoord_t	s = 0;
		for( size_t k = 0; k != dsize; ++k )
			s += c[i+coff+k*cstride]*d[k];
		if( fabs( (b[i+boff] + s) - a[i+aoff] ) > eps )
		{
			ret = false;
		}
	}

	e->free_buffer( mem_a );
	e->free_buffer( mem_b );
	e->free_buffer( mem_c );
	e->free_buffer( mem_d );

	return ret;
}

bool test_fmaddn3( nbody_engine* e, size_t dsize )
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	const size_t			size = e->problem_size();
	const size_t			cstride = size;
	size_t					aoff = 0;
	size_t					coff = 0;
	std::vector<nbcoord_t>	a( e->problem_size() + aoff );
	std::vector<nbcoord_t>	c( e->problem_size()*dsize + coff );
	std::vector<nbcoord_t>	d( dsize );
	nbody_engine::memory*	mem_a = e->create_buffer( a.size()*sizeof( nbcoord_t ) );
	nbody_engine::memory*	mem_c = e->create_buffer( c.size()*sizeof( nbcoord_t ) );
	nbody_engine::memory*	mem_d = e->create_buffer( d.size()*sizeof( nbcoord_t ) );

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

	e->write_buffer( mem_a, a.data() );
	e->write_buffer( mem_c, c.data() );
	e->write_buffer( mem_d, d.data() );

	//! a[i+aoff] = b[i+boff] + sum( c[i+coff+k*cstride]*d[k], k=[0...dsize) )
	e->fmaddn( mem_a, NULL, mem_c, mem_d, cstride, aoff, 0, coff, dsize );

	e->read_buffer( a.data(), mem_a );

	bool	ret = true;
	for( size_t i = 0; i != size; ++i )
	{
		nbcoord_t	s = 0;
		for( size_t k = 0; k < dsize; ++k )
		{
			s += c[i+coff+k*cstride]*d[k];
		}
		if( fabs( s - a[i+aoff] ) > eps )
		{
			ret = false;
		}
	}

	e->free_buffer( mem_a );
	e->free_buffer( mem_c );
	e->free_buffer( mem_d );

	return ret;
}

bool test_fmaxabs( nbody_engine* e )
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	std::vector<nbcoord_t>	a( e->problem_size() );

	if( a.empty() )
	{
		return false;
	}

	nbody_engine::memory*	mem_a = e->create_buffer( sizeof(nbcoord_t)*a.size() );
	nbcoord_t				result = 2878767678687;//Garbage

	for( size_t n = 0; n != a.size(); ++n )
	{
		a[n] = rand() % 10000;
	}

	e->write_buffer( mem_a, a.data() );

	//! @result = max( fabs(a[k]), k=[0...asize) )
	e->fmaxabs( mem_a, result );

	std::transform( a.begin(), a.end(), a.begin(), fabs );
	nbcoord_t testmax = *std::max_element( a.begin(), a.end() );

	bool	ret = ( fabs( result - testmax ) < eps );

	e->free_buffer( mem_a );

	return ret;
}


class test_nbody_engine : public QObject
{
	Q_OBJECT

	nbody_data		data;
	nbody_engine*	e;
public:
	test_nbody_engine( nbody_engine* e );
	~test_nbody_engine();

private slots:
	void initTestCase();
	void cleanupTestCase();
	void test_mem();
	void test_memcpy();
	void test_fmadd1();
	void test_fmadd2();
	void test_fmaddn1();
	void test_fmaddn2();
	void test_fmaddn3();
	void test_fmaxabs();
};

test_nbody_engine::test_nbody_engine( nbody_engine* _e ) :
	e( _e )
{
}

test_nbody_engine::~test_nbody_engine()
{
	delete e;
}

void test_nbody_engine::initTestCase()
{
	nbcoord_t				box_size = 100;

	qDebug() << "Engine" << e->type_name();

	data.make_universe( 64, box_size, box_size, box_size );
	e->init( &data );
}

void test_nbody_engine::cleanupTestCase()
{

}

void test_nbody_engine::test_mem()
{
	QVERIFY( ::test_mem( e ) );
}

void test_nbody_engine::test_memcpy()
{
	QVERIFY( ::test_memcpy( e ) );
}

void test_nbody_engine::test_fmadd1()
{
	QVERIFY( ::test_fmadd1( e ) );
}

void test_nbody_engine::test_fmadd2()
{
	QVERIFY( ::test_fmadd2( e ) );
}

void test_nbody_engine::test_fmaddn1()
{
	QVERIFY( ::test_fmaddn1( e, 1 ) );
	QVERIFY( ::test_fmaddn1( e, 3 ) );
	QVERIFY( ::test_fmaddn1( e, 7 ) );
}

void test_nbody_engine::test_fmaddn2()
{
	QVERIFY( ::test_fmaddn2( e, 1 ) );
	QVERIFY( ::test_fmaddn2( e, 3 ) );
	QVERIFY( ::test_fmaddn2( e, 7 ) );
}

void test_nbody_engine::test_fmaddn3()
{
	QVERIFY( ::test_fmaddn3( e, 1 ) );
	QVERIFY( ::test_fmaddn3( e, 3 ) );
	QVERIFY( ::test_fmaddn3( e, 7 ) );
}

void test_nbody_engine::test_fmaxabs()
{
	QVERIFY( ::test_fmaxabs( e ) );
}

int main(int argc, char *argv[])
{
	int res = 0;

	{
		test_nbody_engine tc1( new nbody_engine_block() );
		res += QTest::qExec( &tc1, argc, argv );
	}
#ifdef HAVE_OPENCL
	{
		test_nbody_engine tc1( new nbody_engine_opencl() );
		res += QTest::qExec( &tc1, argc, argv );
	}
#endif

	{
		test_nbody_engine tc1( new nbody_engine_openmp() );
		res += QTest::qExec( &tc1, argc, argv );
	}

	{
		test_nbody_engine tc3( new nbody_engine_simple() );
		res += QTest::qExec( &tc3, argc, argv );
	}

    return res;
}

#include "test_nbody_engine.moc"
