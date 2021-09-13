#include <QString>
#include <QFileInfo>
#include <QtTest>

#include "nbody_solvers.h"
#include "nbody_engines.h"
#include "summation.h"


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
	void butcher_table_check();
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
	m_e->print_info();
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
	QVERIFY(expected.is_equal(m_data, 1e-12));
}

void test_nbody_solver::butcher_table_check()
{
	nbody_solver_rk_butcher*	rk_butcher =
		dynamic_cast<nbody_solver_rk_butcher*>(m_s);
	if(rk_butcher == nullptr)
	{
		return;
	}
	const nbody_butcher_table*	table = rk_butcher->table();
	QVERIFY(table != nullptr);

	nbcoord_t	b1 = 0_f;
	nbcoord_t	b2 = 0_f;
	int			zero_count = 0;
	for(size_t i = 0; i != table->get_steps(); ++i)
	{
		b1 += table->get_b1()[i];
		b2 += table->get_b2()[i];
		if((table->get_b1()[i] - table->get_b2()[i]) == 0_f)
		{
			zero_count++;
		}
	}
	nbcoord_t	eps(10 * std::numeric_limits<nbcoord_t>::epsilon());
	qDebug() << "(b1 - 1_f) =" << b1 - 1_f << "eps =" << eps;
	qDebug() << "(b2 - 1_f) =" << b2 - 1_f << "eps =" << eps;
	qDebug() << "zero_count(b2 - b2) =" << zero_count << "of" << table->get_steps();
	QVERIFY(fabs(b1 - 1_f) < eps);
	QVERIFY(fabs(b2 - 1_f) < eps);

	size_t total = 0;
	zero_count = 0;
	for(size_t i = 0; i != table->get_steps(); ++i)
	{
		nbcoord_t	a_absmax = 0_f;
		nbcoord_t	a_sum = 0_f;
		nbcoord_t	a_corr = 0_f;
		nbcoord_t	c = table->get_c()[i];
		size_t		jmax = (table->is_implicit() ? table->get_steps() : i);
		if(jmax == 0) { a_absmax = 1_f; }
		for(size_t j = 0; j != jmax; ++j)
		{
			nbcoord_t a = table->get_a()[i][j];
			a_sum = summation_k(a_sum, a, a_corr);
			a_absmax = std::max(a_absmax, static_cast<nbcoord_t>(fabs(a)));
			if(a == 0_f)
			{
				zero_count++;
			}
		}
		total += jmax;
		qDebug() << i << "Sum{a[i]} =" << a_sum << "c = " << c
				 << "(Sum{a[i]} - c[i]) =" << (a_sum - c)
				 << "eps =" << eps << "max(|a|) =" << a_absmax;
		QVERIFY(fabs(a_sum - c) / a_absmax < eps);
	}
	qDebug() << "zero_count(a) =" << zero_count << "of" << total;
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
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "bs"}, {"max_level", 4}, {"min_step", 1e-5}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "bulirsch-stoer");
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
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "midpoint-st"}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "midpoint-st");
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
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "rkdp"}, {"correction", false}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "rkdp");
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "rkdp"}, {"correction", true}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "rkdp-corr");
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
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "rkfeagin10"}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "rkfeagin10");
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "rkfeagin10"}, {"correction", true}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "rkfeagin10-corr");
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "rkfeagin12"}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "rkfeagin12");
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"solver", "rkfeagin14"}}));
		test_nbody_solver	tc1(argv[0], new nbody_engine_active(), nbody_create_solver(param), "rkfeagin14");
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
#ifdef HAVE_OPENCL
	{
		QVariantMap			sparam(std::map<QString, QVariant>({{"solver", "euler"}}));
		QVariantMap			eparam(std::map<QString, QVariant>({{"engine", "opencl"}, {"block_size", 4}}));
		test_nbody_solver	tc1(argv[0], nbody_create_engine(eparam), nbody_create_solver(sparam), "euler");
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			sparam(std::map<QString, QVariant>({{"solver", "euler"}}));
		QVariantMap			eparam(std::map<QString, QVariant>({{"engine", "opencl"}, {"block_size", 4}, {"device", "0:0,0"}}));
		test_nbody_solver	tc1(argv[0], nbody_create_engine(eparam), nbody_create_solver(sparam), "euler");
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			sparam(std::map<QString, QVariant>({{"solver", "euler"}}));
		QVariantMap			eparam(std::map<QString, QVariant>({{"engine", "opencl_bh"},
			{"block_size", 4},
			{"distance_to_node_radius_ratio", 1e8}}));
		test_nbody_solver	tc1(argv[0], nbody_create_engine(eparam), nbody_create_solver(sparam), "euler");
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			sparam(std::map<QString, QVariant>({{"solver", "euler"}}));
		QVariantMap			eparam(std::map<QString, QVariant>({{"engine", "opencl_bh"},
			{"block_size", 4},
			{"distance_to_node_radius_ratio", 1e8},
			{"device", "0:0,0"}}));
		test_nbody_solver	tc1(argv[0], nbody_create_engine(eparam), nbody_create_solver(sparam), "euler");
		res += QTest::qExec(&tc1, argc, argv);
	}
#endif //HAVE_OPENCL
	return res;
}

#include "test_nbody_solver.moc"
