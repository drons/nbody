#include <QApplication>
#include <QtTest>
#include <QDebug>
#include <limits>
#include <omp.h>

#include "nbody_engines.h"
#include "nbody_extrapolator.h"

class test_nbody_extrapolator : public QObject
{
	Q_OBJECT
	nbody_data		m_data;
	nbody_engine*	m_e;
	size_t			m_problem_size;
public:
	explicit test_nbody_extrapolator(nbody_engine* e, size_t problen_size = 1);
	~test_nbody_extrapolator();

private slots:
	void initTestCase();
	void cleanupTestCase();
	void extrapolate();
};

test_nbody_extrapolator::test_nbody_extrapolator(nbody_engine* e, size_t problen_size) :
	m_e(e), m_problem_size(problen_size)
{
}

test_nbody_extrapolator::~test_nbody_extrapolator()
{
	delete m_e;
}

void test_nbody_extrapolator::initTestCase()
{
	qDebug() << "Engine" << m_e->type_name() << "Problem size" << m_problem_size;
	m_e->print_info();
	m_data.resize(m_problem_size);
	m_e->init(&m_data);
}

void test_nbody_extrapolator::cleanupTestCase()
{
}

void test_nbody_extrapolator::extrapolate()
{
	nbody_engine::memory*		y = m_e->create_buffer(m_e->problem_size() *
													   sizeof(nbcoord_t));
	std::vector<size_t>			sc{2, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128};
	nbody_extrapolator_berrut	ex(m_e, 1, sc);
	auto	func = [](nbcoord_t x) {return 1_f / (1_f + x * x) + 1_f;};

	nbcoord_t		last_error_estimation = 1e30;
	nbcoord_t		last_error_absolute = 1e30;
	for(size_t level = 0; level != sc.size(); ++level)
	{
		std::vector<nbcoord_t>	host_y(m_e->problem_size(), func(sc[level]));
		m_e->write_buffer(y, host_y.data());
		ex.update_table(level, y);
		if(level < 2)
		{
			continue;
		}
		nbcoord_t	err = ex.estimate_error(level);
		qDebug() << "level =" << level << "err =" << err;
		QVERIFY(last_error_estimation >= err);
		last_error_estimation = err;
		ex.extrapolate(level, y);
		m_e->read_buffer(host_y.data(), y);

		nbcoord_t expected_value = func(1e15_f);
		nbcoord_t error_absolute = fabs(host_y[0] - expected_value);
		qDebug() << "func(inf) =" << expected_value
				 << "extrapolate =" << host_y[0]
				 << "absdiff =" << error_absolute << err;
		QVERIFY(error_absolute < err);
		QVERIFY(last_error_absolute >= error_absolute);
		last_error_absolute = error_absolute;
	}

	m_e->free_buffer(y);
}



int main(int argc, char* argv[])
{
	int res = 0;
	{
		QVariantMap			param(std::map<QString, QVariant>({{"engine", "simple"}}));
		test_nbody_extrapolator	tc1(nbody_create_engine(param));
		res += QTest::qExec(&tc1, argc, argv);
	}
	return res;
}

#include "test_nbody_extrapolator.moc"
