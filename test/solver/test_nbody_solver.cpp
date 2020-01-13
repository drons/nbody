#include <QString>
#include <QFileInfo>
#include <QtTest>

#include "nbody_solvers.h"
#include "nbody_engines.h"


class test_nbody_solver : public QObject
{
	Q_OBJECT
	QString			m_apppath;
	QString			m_test_name;
	nbody_data		m_data;
	nbody_engine*	m_e;
	nbody_solver*	m_s;
public:
	test_nbody_solver(const QString& apppath, nbody_engine* m_e, nbody_solver* m_s, const QString& test_name);
	~test_nbody_solver();
private Q_SLOTS:
	void initTestCase();
	void cleanupTestCase();
	void run();
};

test_nbody_solver::test_nbody_solver(const QString& apppath, nbody_engine* _e,
									 nbody_solver* _s, const QString& test_name) :
	m_apppath(QFileInfo(apppath).absolutePath()),
	m_test_name(test_name),
	m_e(_e),
	m_s(_s)
{
}

test_nbody_solver::~test_nbody_solver()
{
	delete m_s;
	delete m_e;
}

void test_nbody_solver::initTestCase()
{
	nbody_solver_rk_butcher*	butcher = dynamic_cast<nbody_solver_rk_butcher*>(m_s);

	if(butcher != NULL)
	{
		butcher->set_max_recursion(1);
		butcher->set_substep_subdivisions(2);
		butcher->set_refine_steps_count(1);
		butcher->set_error_threshold(1e-5);
	}

	qDebug() << "Solver =" << m_s->type_name() << "engine" << m_e->type_name();
	m_s->print_info();
	QString	data_path(m_apppath + "/../data/initial_state.txt");
	QVERIFY(m_data.load(data_path));

//	nbcoord_t	box_size = 100;
//	data.make_universe(8, box_size, box_size, box_size);
//	data.save(data_path);

	m_e->init(&m_data);
	m_s->set_time_step(1e-3, 3e-2);
	m_s->set_engine(m_e);
}

void test_nbody_solver::cleanupTestCase()
{
}

void test_nbody_solver::run()
{
	m_data.print_statistics(m_e);
	m_s->run(&m_data, NULL, 0.3, 0, 0.0);
	m_data.print_statistics(m_e);

	nbody_data	expected;
	QString		data_path(m_apppath + "/../data/" + m_test_name + ".txt");
	//m_data.save(data_path);
	QVERIFY(expected.load(data_path));
	QVERIFY(expected.is_equal(m_data, 1e-13));
}

typedef nbody_engine_simple	nbody_engine_active;

int main(int argc, char* argv[])
{
	int res = 0;

	{
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "invalid"}}));
		nbody_solver*		s(nbody_create_solver(param));
		if(s != NULL)
		{
			qDebug() << "Created solver with invalid type" << param;
			res += 1;
			delete s;
		}
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "adams"}, {"starter_solver", "invalid"}}));
		nbody_solver*		s(nbody_create_solver(param));
		if(s != NULL)
		{
			qDebug() << "Created solver with invalid type" << param;
			res += 1;
			delete s;
		}
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "adams"}, {"rank", 5}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "adams5");
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "adams"}, {"rank", 5}, {"correction", true}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "adams5-corr");
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "euler"}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "euler");
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "midpoint"}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "midpoint");
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "rk4"}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "rk4");
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "rkck"}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "rkck");
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "rkdp"}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "rkdp");
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "rkdverk"}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "rkdverk");
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "rkf"}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "rkf");
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "rkgl"}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "rkgl");
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "rklc"}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "rklc");
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "trapeze"}, {"refine_steps_count", 2}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "trapeze2");
		res += QTest::qExec(&tc1, argc, argv);
	}
	return res;
}

#include "test_nbody_solver.moc"
