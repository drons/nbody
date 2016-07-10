#include <QString>
#include <QtTest>

#include "nbody_solvers.h"
#include "nbody_engines.h"

class test_nbody_solver : public QObject
{
	Q_OBJECT

	nbody_data		data;
	nbody_engine*	e;
	nbody_solver*	s;
public:
	test_nbody_solver( nbody_engine* e, nbody_solver* s );
	~test_nbody_solver();
private Q_SLOTS:
	void initTestCase();
	void cleanupTestCase();
	void run();
};

test_nbody_solver::test_nbody_solver( nbody_engine* _e, nbody_solver* _s ) :
	e( _e ), s( _s )
{

}

test_nbody_solver::~test_nbody_solver()
{
	delete s;
	delete e;
}

void test_nbody_solver::initTestCase()
{
	nbcoord_t				box_size = 100;

	data.make_universe( box_size, box_size, box_size );

	e->init( &data );
	s->set_time_step( 1e-3, 1e-2 );
	s->set_engine( e );
}

void test_nbody_solver::cleanupTestCase()
{
}

void test_nbody_solver::run()
{
	s->run( &data, 0.3, 0, 0.1 );
}

typedef nbody_engine_simple	nbody_engine_active;

int main(int argc, char *argv[])
{
	int res = 0;

	{
		test_nbody_solver tc1( new nbody_engine_active(), new nbody_solver_adams() );
		res += QTest::qExec( &tc1, argc, argv );
	}
	{
		test_nbody_solver tc1( new nbody_engine_active(), new nbody_solver_euler() );
		res += QTest::qExec( &tc1, argc, argv );
	}
	{
		test_nbody_solver tc1( new nbody_engine_active(), new nbody_solver_rk4() );
		res += QTest::qExec( &tc1, argc, argv );
	}
	{
		test_nbody_solver tc1( new nbody_engine_active(), new nbody_solver_rkck() );
		res += QTest::qExec( &tc1, argc, argv );
	}
	{
		test_nbody_solver tc1( new nbody_engine_active(), new nbody_solver_rkdp() );
		res += QTest::qExec( &tc1, argc, argv );
	}
	{
		test_nbody_solver tc1( new nbody_engine_active(), new nbody_solver_rkf() );
		res += QTest::qExec( &tc1, argc, argv );
	}
	{
		test_nbody_solver tc1( new nbody_engine_active(), new nbody_solver_rkgl() );
		res += QTest::qExec( &tc1, argc, argv );
	}
	{
		test_nbody_solver tc1( new nbody_engine_active(), new nbody_solver_rklc() );
		res += QTest::qExec( &tc1, argc, argv );
	}
	{
		test_nbody_solver tc1( new nbody_engine_active(), new nbody_solver_trapeze() );
		res += QTest::qExec( &tc1, argc, argv );
	}
	return res;
}

#include "test_nbody_solver.moc"