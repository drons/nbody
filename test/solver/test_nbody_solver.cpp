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
	test_nbody_solver(nbody_engine* e, nbody_solver* s);
	~test_nbody_solver();
private Q_SLOTS:
	void initTestCase();
	void cleanupTestCase();
	void run();
};

test_nbody_solver::test_nbody_solver(nbody_engine* _e, nbody_solver* _s) :
	e(_e), s(_s)
{

}

test_nbody_solver::~test_nbody_solver()
{
	delete s;
	delete e;
}

void test_nbody_solver::initTestCase()
{
	nbcoord_t					box_size = 100;
	nbody_solver_rk_butcher*	butcher = dynamic_cast<nbody_solver_rk_butcher*>(s);

	if(butcher != NULL)
	{
		butcher->set_max_recursion(1);
		butcher->set_substep_subdivisions(2);
		butcher->set_refine_steps_count(1);
		butcher->set_error_threshold(1e-5);
	}

	qDebug() << "Solver =" << s->type_name() << "engine" << e->type_name();
	data.make_universe(64, box_size, box_size, box_size);

	e->init(&data);
	s->set_time_step(1e-3, 3e-2);
	s->set_engine(e);
}

void test_nbody_solver::cleanupTestCase()
{
}

void test_nbody_solver::run()
{
	data.print_statistics(e);
	s->run(&data, NULL, 0.3, 0, 0.0);
	data.print_statistics(e);
}

typedef nbody_engine_simple	nbody_engine_active;

int main(int argc, char* argv[])
{
	int res = 0;

	{
		QVariantMap			param{{"solver", "adams"}, {"rank", 5}};
		test_nbody_solver	tc1(new nbody_engine_active(), nbody_create_solver(param));
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param{{"solver", "euler"}};
		test_nbody_solver	tc1(new nbody_engine_active(), nbody_create_solver(param));
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param{{"solver", "rk4"}};
		test_nbody_solver	tc1(new nbody_engine_active(), nbody_create_solver(param));
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param{{"solver", "rkck"}};
		test_nbody_solver	tc1(new nbody_engine_active(), nbody_create_solver(param));
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param{{"solver", "rkdp"}};
		test_nbody_solver	tc1(new nbody_engine_active(), nbody_create_solver(param));
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param{{"solver", "rkf"}};
		test_nbody_solver	tc1(new nbody_engine_active(), nbody_create_solver(param));
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param{{"solver", "rkgl"}};
		test_nbody_solver	tc1(new nbody_engine_active(), nbody_create_solver(param));
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param{{"solver", "rklc"}};
		test_nbody_solver	tc1(new nbody_engine_active(), nbody_create_solver(param));
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param{{"solver", "trapeze"}, {"refine_steps_count", 2}};
		test_nbody_solver	tc1(new nbody_engine_active(), nbody_create_solver(param));
		res += QTest::qExec(&tc1, argc, argv);
	}
	return res;
}

#include "test_nbody_solver.moc"
