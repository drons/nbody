#include <QString>
#include <QtTest>

#include "nbody_solvers.h"
#include "nbody_engines.h"
#include "summation.h"

class test_nbody_solvers_equality : public QObject
{
	Q_OBJECT

	nbody_data		data;
	nbody_engine*	e1;
	nbody_solver*	s1;
	nbody_engine*	e2;
	nbody_solver*	s2;
public:
	test_nbody_solvers_equality(nbody_engine* _e1, nbody_solver* _s1, nbody_engine* _e2, nbody_solver* _s2);
	~test_nbody_solvers_equality();
private:
	bool check_y(nbcoord_t max_max, nbcoord_t max_mean);
private Q_SLOTS:
	void initTestCase();
	void cleanupTestCase();
	void run();
};

test_nbody_solvers_equality::test_nbody_solvers_equality(nbody_engine* _e1, nbody_solver* _s1, nbody_engine* _e2,
														 nbody_solver* _s2) :
	e1(_e1), s1(_s1), e2(_e2), s2(_s2)
{
}

test_nbody_solvers_equality::~test_nbody_solvers_equality()
{
	delete s1;
	delete s2;
	delete e1;
	delete e2;
}

bool test_nbody_solvers_equality::check_y(nbcoord_t max_max, nbcoord_t max_mean)
{
	std::vector<nbcoord_t>	y1(e1->problem_size());
	std::vector<nbcoord_t>	y2(e2->problem_size());

	e1->read_buffer(y1.data(), e1->get_y());
	e2->read_buffer(y2.data(), e2->get_y());

	std::transform(y1.begin(), y1.end(), y2.begin(), y1.begin(), std::minus<nbcoord_t>());
	std::transform(y1.begin(), y1.end(), y1.begin(), y1.begin(), std::multiplies<nbcoord_t>());

	nbcoord_t	max_delta = sqrt(*std::max_element(y1.begin(), y1.end()));
	nbcoord_t	min_delta = sqrt(*std::min_element(y1.begin(), y1.end()));
	nbcoord_t	mean_delta = sqrt(summation<nbcoord_t, const nbcoord_t*>(y1.data(),
																		 y1.size())) / static_cast<nbcoord_t>(y1.size());

	qDebug() << "max_delta" << max_delta << "min_delta" << min_delta << "mean_delta" << mean_delta;
	qDebug() << "max_max" << max_max << "max_mean" << max_mean;
	return (max_delta < max_max) && (mean_delta < max_mean);
}

void test_nbody_solvers_equality::initTestCase()
{
	nbcoord_t					box_size = 100;

	qDebug() << "#1 Solver =" << s1->type_name() << "engine" << e1->type_name();
	qDebug() << "#2 Solver =" << s2->type_name() << "engine" << e2->type_name();

	data.make_universe(64, box_size, box_size, box_size);

	QVERIFY(e1->init(&data));
	s1->set_time_step(1e-3, 1e-2);
	s1->set_engine(e1);

	QVERIFY(e2->init(&data));
	s2->set_time_step(1e-3, 1e-2);
	s2->set_engine(e2);
}

void test_nbody_solvers_equality::cleanupTestCase()
{
}

void test_nbody_solvers_equality::run()
{
	QVERIFY(e1->problem_size() == e2->problem_size());

	QVERIFY(check_y(1e-18, 1e-18));

	const int MAX_STEPS = 7;

	for(int i = 0; i != MAX_STEPS; ++i)
	{
		s1->advise(s1->get_max_step());
		s2->advise(s2->get_max_step());
	}

	QVERIFY(check_y(1e-13, 1e-16));
}

typedef nbody_engine_simple	nbody_engine_active;

int main(int argc, char* argv[])
{
	int res = 0;

	{
		test_nbody_solvers_equality tc1(new nbody_engine_active(), new nbody_solver_rk4(),
										new nbody_engine_active(), new nbody_solver_rk_butcher(new nbody_butcher_table_rk4()));
		res += QTest::qExec(&tc1, argc, argv);
	}

	{
		test_nbody_solvers_equality tc1(new nbody_engine_active(), new nbody_solver_euler(),
										new nbody_engine_active(), new nbody_solver_adams(new nbody_solver_euler(), 1));
		res += QTest::qExec(&tc1, argc, argv);
	}

	return res;
}

#include "test_nbody_solvers_equality.moc"
