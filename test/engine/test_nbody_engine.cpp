#include <QApplication>
#include <QtTest>
#include <QDebug>
#include <limits>
#include <omp.h>

#include "nbody_engines.h"

bool test_mem(nbody_engine* e)
{
	nbody_engine::memory*	mem = e->create_buffer(1024);

	if(mem == NULL)
	{
		return false;
	}
	e->free_buffer(mem);

	return true;
}

bool test_memcpy(nbody_engine* e)
{
	const size_t			cnt = 8;
	const size_t			size = sizeof(nbcoord_t) * cnt;
	nbcoord_t				data[cnt] = { 0, 1, 2, 3, 4, 5, 6, 7 };
	nbcoord_t				x[cnt] = {0};
	nbody_engine::memory*	mem = e->create_buffer(size);
	nbody_engine::memory*	submem = e->create_buffer(size / 2);

	e->write_buffer(mem, data);
	e->read_buffer(x, mem);

	bool	ret = (0 == memcmp(data, x, size));

	e->free_buffer(mem);
	e->free_buffer(submem);

	return ret;
}

bool test_copy_buffer(nbody_engine* e)
{
	const size_t			cnt = e->problem_size();
	size_t					off1  = 33;
	size_t					off2 = 44;
	std::vector<nbcoord_t>	data1;
	std::vector<nbcoord_t>	data2;
	nbody_engine::memory*	mem1 = e->create_buffer((cnt + off1) * sizeof(nbcoord_t));
	nbody_engine::memory*	mem2 = e->create_buffer((cnt + off2) * sizeof(nbcoord_t));

	data1.resize(cnt + off1);
	data2.resize(cnt + off2);

	for(size_t i = 0; i != data1.size(); ++i)
	{
		data1[i] = static_cast<nbcoord_t>(i);
	}

	e->write_buffer(mem1, data1.data());
	e->copy_buffer(mem2, mem1, off2, off1);
	e->read_buffer(data2.data(), mem2);

	bool	ret = (0 == memcmp(data1.data() + off1, data2.data() + off2, cnt * sizeof(nbcoord_t)));

	e->free_buffer(mem1);
	e->free_buffer(mem2);

	return ret;
}
bool test_fmadd1(nbody_engine* e)
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	const size_t			size = sizeof(nbcoord_t) * e->problem_size();
	std::vector<nbcoord_t>	a(e->problem_size());
	std::vector<nbcoord_t>	a_res(e->problem_size());
	std::vector<nbcoord_t>	b(e->problem_size());
	nbcoord_t				c = 5;
	nbody_engine::memory*	mem_a = e->create_buffer(size);
	nbody_engine::memory*	mem_b = e->create_buffer(size);

	for(size_t n = 0; n != a.size(); ++n)
	{
		a[n] = rand() % 10000;
		b[n] = rand() % 10000;
	}

	e->write_buffer(mem_a, a.data());
	e->write_buffer(mem_b, b.data());

	//! a[i] += b[i]*c
	e->fmadd_inplace(mem_a, mem_b, c);

	e->read_buffer(a_res.data(), mem_a);

	bool	ret = true;
	for(size_t i = 0; i != a.size(); ++i)
	{
		if(fabs((a[i] + c * b[i]) - a_res[i]) > eps)
		{
			ret = false;
		}
	}

	e->free_buffer(mem_a);
	e->free_buffer(mem_b);

	return ret;
}

bool test_fmadd2(nbody_engine* e)
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	const size_t			size = e->problem_size();
	size_t					aoff = 33;
	size_t					boff = 44;
	size_t					coff = 55;
	std::vector<nbcoord_t>	a(e->problem_size() + aoff);
	std::vector<nbcoord_t>	b(e->problem_size() + boff);
	std::vector<nbcoord_t>	c(e->problem_size() + coff);
	nbcoord_t				d = 5;
	nbody_engine::memory*	mem_a = e->create_buffer(a.size() * sizeof(nbcoord_t));
	nbody_engine::memory*	mem_b = e->create_buffer(b.size() * sizeof(nbcoord_t));
	nbody_engine::memory*	mem_c = e->create_buffer(c.size() * sizeof(nbcoord_t));

	for(size_t n = 0; n != a.size(); ++n)
	{
		a[n] = rand() % 10000;
	}
	for(size_t n = 0; n != b.size(); ++n)
	{
		b[n] = rand() % 10000;
	}
	for(size_t n = 0; n != c.size(); ++n)
	{
		c[n] = rand() % 10000;
	}

	e->write_buffer(mem_a, a.data());
	e->write_buffer(mem_b, b.data());
	e->write_buffer(mem_c, c.data());

	//! a[i+aoff] = b[i+boff] + c[i+coff]*d
	e->fmadd(mem_a, mem_b, mem_c, d, aoff, boff, coff);

	e->read_buffer(a.data(), mem_a);

	bool	ret = true;
	for(size_t i = 0; i != size; ++i)
	{
		if(fabs((b[i + boff] + c[i + coff]*d) - a[i + aoff]) > eps)
		{
			ret = false;
		}
	}

	e->free_buffer(mem_a);
	e->free_buffer(mem_b);
	e->free_buffer(mem_c);

	return ret;
}

bool test_fmaddn1(nbody_engine* e, size_t csize)
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	const size_t			size = e->problem_size();
	const size_t			bstride = size;
	size_t					aoff = 33;
	size_t					boff = 44;
	std::vector<nbcoord_t>	a(e->problem_size() + aoff);
	std::vector<nbcoord_t>	a_res(e->problem_size() + aoff);
	std::vector<nbcoord_t>	b(e->problem_size()*csize + boff);
	std::vector<nbcoord_t>	c(csize);
	nbody_engine::memory*	mem_a = e->create_buffer(a.size() * sizeof(nbcoord_t));
	nbody_engine::memory*	mem_b = e->create_buffer(b.size() * sizeof(nbcoord_t));
	nbody_engine::memory*	mem_c = e->create_buffer(c.size() * sizeof(nbcoord_t));

	for(size_t n = 0; n != a.size(); ++n)
	{
		a[n] = rand() % 10000;
	}
	for(size_t n = 0; n != b.size(); ++n)
	{
		b[n] = rand() % 10000;
	}
	for(size_t n = 0; n != c.size(); ++n)
	{
		c[n] = rand() % 10000;
	}

	e->write_buffer(mem_a, a.data());
	e->write_buffer(mem_b, b.data());
	e->write_buffer(mem_c, c.data());

	//! a[i+aoff] += sum( b[i+boff+k*bstride]*c[k], k=[0...csize) )
	e->fmaddn_inplace(mem_a, mem_b, mem_c, bstride, aoff, boff, csize);

	e->read_buffer(a_res.data(), mem_a);

	bool	ret = true;
	for(size_t i = 0; i != size; ++i)
	{
		nbcoord_t	s = 0;
		for(size_t k = 0; k != csize; ++k)
		{
			s += b[i + boff + k * bstride] * c[k];
		}
		if(fabs((a[i + aoff] + s) - a_res[i + aoff]) > eps)
		{
			ret = false;
		}
	}

	e->free_buffer(mem_a);
	e->free_buffer(mem_b);
	e->free_buffer(mem_c);

	return ret;
}

bool test_fmaddn2(nbody_engine* e, size_t dsize)
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	const size_t			size = e->problem_size();
	const size_t			cstride = size;
	size_t					aoff = 33;
	size_t					boff = 44;
	size_t					coff = 55;
	std::vector<nbcoord_t>	a(e->problem_size() + aoff);
	std::vector<nbcoord_t>	b(e->problem_size() + boff);
	std::vector<nbcoord_t>	c(e->problem_size()*dsize + coff);
	std::vector<nbcoord_t>	d(dsize);
	nbody_engine::memory*	mem_a = e->create_buffer(a.size() * sizeof(nbcoord_t));
	nbody_engine::memory*	mem_b = e->create_buffer(b.size() * sizeof(nbcoord_t));
	nbody_engine::memory*	mem_c = e->create_buffer(c.size() * sizeof(nbcoord_t));
	nbody_engine::memory*	mem_d = e->create_buffer(d.size() * sizeof(nbcoord_t));

	for(size_t n = 0; n != a.size(); ++n)
	{
		a[n] = rand() % 10000;
	}
	for(size_t n = 0; n != b.size(); ++n)
	{
		b[n] = rand() % 10000;
	}
	for(size_t n = 0; n != c.size(); ++n)
	{
		c[n] = rand() % 10000;
	}
	for(size_t n = 0; n != d.size(); ++n)
	{
		d[n] = rand() % 10000;
	}

	e->write_buffer(mem_a, a.data());
	e->write_buffer(mem_b, b.data());
	e->write_buffer(mem_c, c.data());
	e->write_buffer(mem_d, d.data());

	//! a[i+aoff] = b[i+boff] + sum( c[i+coff+k*cstride]*d[k], k=[0...dsize) )
	e->fmaddn(mem_a, mem_b, mem_c, mem_d, cstride, aoff, boff, coff, dsize);

	e->read_buffer(a.data(), mem_a);

	bool	ret = true;
	for(size_t i = 0; i != size; ++i)
	{
		nbcoord_t	s = 0;
		for(size_t k = 0; k != dsize; ++k)
		{
			s += c[i + coff + k * cstride] * d[k];
		}
		if(fabs((b[i + boff] + s) - a[i + aoff]) > eps)
		{
			ret = false;
		}
	}

	e->free_buffer(mem_a);
	e->free_buffer(mem_b);
	e->free_buffer(mem_c);
	e->free_buffer(mem_d);

	return ret;
}

bool test_fmaddn3(nbody_engine* e, size_t dsize)
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	const size_t			size = e->problem_size();
	const size_t			cstride = size;
	size_t					aoff = 0;
	size_t					coff = 0;
	std::vector<nbcoord_t>	a(e->problem_size() + aoff);
	std::vector<nbcoord_t>	c(e->problem_size()*dsize + coff);
	std::vector<nbcoord_t>	d(dsize);
	nbody_engine::memory*	mem_a = e->create_buffer(a.size() * sizeof(nbcoord_t));
	nbody_engine::memory*	mem_c = e->create_buffer(c.size() * sizeof(nbcoord_t));
	nbody_engine::memory*	mem_d = e->create_buffer(d.size() * sizeof(nbcoord_t));

	for(size_t n = 0; n != a.size(); ++n)
	{
		a[n] = rand() % 10000;
	}
	for(size_t n = 0; n != c.size(); ++n)
	{
		c[n] = rand() % 10000;
	}
	for(size_t n = 0; n != d.size(); ++n)
	{
		d[n] = rand() % 10000;
	}

	e->write_buffer(mem_a, a.data());
	e->write_buffer(mem_c, c.data());
	e->write_buffer(mem_d, d.data());

	//! a[i+aoff] = b[i+boff] + sum( c[i+coff+k*cstride]*d[k], k=[0...dsize) )
	e->fmaddn(mem_a, NULL, mem_c, mem_d, cstride, aoff, 0, coff, dsize);

	e->read_buffer(a.data(), mem_a);

	bool	ret = true;
	for(size_t i = 0; i != size; ++i)
	{
		nbcoord_t	s = 0;
		for(size_t k = 0; k < dsize; ++k)
		{
			s += c[i + coff + k * cstride] * d[k];
		}
		if(fabs(s - a[i + aoff]) > eps)
		{
			ret = false;
		}
	}

	e->free_buffer(mem_a);
	e->free_buffer(mem_c);
	e->free_buffer(mem_d);

	return ret;
}

bool test_fmaxabs(nbody_engine* e)
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	std::vector<nbcoord_t>	a(e->problem_size());

	if(a.empty())
	{
		return false;
	}

	nbody_engine::memory*	mem_a = e->create_buffer(sizeof(nbcoord_t) * a.size());
	nbcoord_t				result = 2878767678687;//Garbage

	for(size_t n = 0; n != a.size(); ++n)
	{
		a[n] = rand() % 10000 - 9000;
	}

	e->write_buffer(mem_a, a.data());

	//! @result = max( fabs(a[k]), k=[0...asize) )
	e->fmaxabs(mem_a, result);

	std::for_each(a.begin(), a.end(), [](nbcoord_t& x) {x = fabs(x);});
	nbcoord_t testmax = *std::max_element(a.begin(), a.end());

	bool	ret = (fabs(result - testmax) < eps);

	e->free_buffer(mem_a);

	return ret;
}

bool test_fcompute(nbody_engine* e, nbody_data* data, const nbcoord_t eps)
{
	nbody_engine_simple		e0;
	e0.init(data);

	std::vector<nbcoord_t>	f0(e0.problem_size());

	{
		double					tbegin = omp_get_wtime();
		nbody_engine::memory*	fbuff;
		fbuff = e0.create_buffer(sizeof(nbcoord_t) * e0.problem_size());
		e0.fcompute(0, e0.get_y(), fbuff, 0, 0);

		e0.read_buffer(f0.data(), fbuff);
		e0.free_buffer(fbuff);
		qDebug() << "Time" << e0.type_name() << omp_get_wtime() - tbegin;
	}


	std::vector<nbcoord_t>	f(e->problem_size());

	{
		double					tbegin = omp_get_wtime();
		nbody_engine::memory*	fbuff;
		fbuff = e->create_buffer(sizeof(nbcoord_t) * e->problem_size());
		e->fcompute(0, e->get_y(), fbuff, 0, 0);

		e->read_buffer(f.data(), fbuff);
		e->free_buffer(fbuff);
		qDebug() << "Time" << e->type_name() << omp_get_wtime() - tbegin;
	}

	bool		ret = true;
	nbcoord_t	total_err = 0;
	nbcoord_t	total_relative_err = 0;
	nbcoord_t	err_smooth = 1e-30;
	size_t		outliers_count = 0;
	for(size_t i = 0; i != f.size(); ++i)
	{
		total_err += fabs(f[i] - f0[i]);
		total_relative_err += 2.0 * (fabs(f[i] - f0[i]) + err_smooth) /
							  (fabs(f[i]) + fabs(f0[i]) + err_smooth);
		if(fabs(f[i] - f0[i]) > eps)
		{
			++outliers_count;
			ret = false;
		}
	}

	if(!ret)
	{
		qDebug() << "Total count:         " << f.size();
		qDebug() << "Total error:         " << total_err;
		qDebug() << "Mean error:          " << total_err / f.size();
		qDebug() << "Total relative error:" << total_relative_err;
		qDebug() << "Outliers count:      " << outliers_count;
	}

	return ret;
}

class test_nbody_engine : public QObject
{
	Q_OBJECT

	nbody_data		m_data;
	nbody_engine*	m_e;
	size_t			m_problem_size;
	nbcoord_t		m_eps;
public:
	test_nbody_engine(nbody_engine* e, size_t problen_size = 64, nbcoord_t eps = 1e-13);
	~test_nbody_engine();

private slots:
	void initTestCase();
	void cleanupTestCase();
	void test_mem();
	void test_memcpy();
	void test_copy_buffer();
	void test_fmadd1();
	void test_fmadd2();
	void test_fmaddn1();
	void test_fmaddn2();
	void test_fmaddn3();
	void test_fmaxabs();
	void test_fcompute();
	void test_negative_branches();
};

test_nbody_engine::test_nbody_engine(nbody_engine* e, size_t problen_size, nbcoord_t eps) :
	m_e(e), m_problem_size(problen_size), m_eps(eps)
{
}

test_nbody_engine::~test_nbody_engine()
{
	delete m_e;
}

void test_nbody_engine::initTestCase()
{
	nbcoord_t				box_size = 100;

	qDebug() << "Engine" << m_e->type_name() << "Problem size" << m_problem_size;

	m_data.make_universe(m_problem_size, box_size, box_size, box_size);
	m_e->init(&m_data);
}

void test_nbody_engine::cleanupTestCase()
{

}

void test_nbody_engine::test_mem()
{
	QVERIFY(::test_mem(m_e));
}

void test_nbody_engine::test_memcpy()
{
	QVERIFY(::test_memcpy(m_e));
}

void test_nbody_engine::test_copy_buffer()
{
	QVERIFY(::test_copy_buffer(m_e));
}

void test_nbody_engine::test_fmadd1()
{
	QVERIFY(::test_fmadd1(m_e));
}

void test_nbody_engine::test_fmadd2()
{
	QVERIFY(::test_fmadd2(m_e));
}

void test_nbody_engine::test_fmaddn1()
{
	QVERIFY(::test_fmaddn1(m_e, 1));
	QVERIFY(::test_fmaddn1(m_e, 3));
	QVERIFY(::test_fmaddn1(m_e, 7));
}

void test_nbody_engine::test_fmaddn2()
{
	QVERIFY(::test_fmaddn2(m_e, 1));
	QVERIFY(::test_fmaddn2(m_e, 3));
	QVERIFY(::test_fmaddn2(m_e, 7));
}

void test_nbody_engine::test_fmaddn3()
{
	QVERIFY(::test_fmaddn3(m_e, 1));
	QVERIFY(::test_fmaddn3(m_e, 3));
	QVERIFY(::test_fmaddn3(m_e, 7));
}

void test_nbody_engine::test_fmaxabs()
{
	QVERIFY(::test_fmaxabs(m_e));
}

void test_nbody_engine::test_fcompute()
{
	QVERIFY(::test_fcompute(m_e, &m_data, m_eps));
}

class nbody_engine_memory_fake : public nbody_engine::memory
{
	size_t m_size;
public:
	nbody_engine_memory_fake(size_t sz) : m_size(sz)
	{
	}
	size_t size() const override
	{
		return m_size;
	}
};

void test_nbody_engine::test_negative_branches()
{
	qDebug() << "fcompute";
	{
		nbody_engine_memory_fake	y(0);
		nbody_engine::memory*		f = m_e->create_buffer(m_e->problem_size());
		m_e->fcompute(0, &y, f, 0, 0);
		m_e->free_buffer(f);
	}
	{
		nbody_engine_memory_fake	f(0);
		nbody_engine::memory*		y = m_e->create_buffer(m_e->problem_size());
		m_e->fcompute(0, y, &f, 0, 0);
		m_e->free_buffer(y);
	}

	qDebug() << "read_buffer";
	{
		nbody_engine_memory_fake	src(0);
		m_e->read_buffer(NULL, &src);
	}

	qDebug() << "write_buffer";
	{
		nbody_engine_memory_fake	dst(0);
		m_e->write_buffer(&dst, NULL);
	}

	qDebug() << "copy_buffer";
	{
		nbody_engine_memory_fake	a(0);
		nbody_engine::memory*		b = m_e->create_buffer(m_e->problem_size());
		m_e->copy_buffer(&a, b, 0, 0);
		m_e->free_buffer(b);
	}
	{
		nbody_engine_memory_fake	b(0);
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		m_e->copy_buffer(a, &b, 0, 0);
		m_e->free_buffer(a);
	}

	qDebug() << "fmadd_inplace";
	{
		nbody_engine_memory_fake	a(0);
		nbody_engine::memory*		b = m_e->create_buffer(m_e->problem_size());
		m_e->fmadd_inplace(&a, b, 1);
		m_e->free_buffer(b);
	}
	{
		nbody_engine_memory_fake	b(0);
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		m_e->fmadd_inplace(a, &b, 1);
		m_e->free_buffer(a);
	}
	qDebug() << "fmadd";
	{
		nbody_engine_memory_fake	a(0);
		nbody_engine::memory*		b = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		c = m_e->create_buffer(m_e->problem_size());
		m_e->fmadd(&a, b, c, 0, 0, 0, 0);
		m_e->free_buffer(b);
		m_e->free_buffer(c);
	}
	{
		nbody_engine_memory_fake	c(0);
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		b = m_e->create_buffer(m_e->problem_size());
		m_e->fmadd(a, b, &c, 0, 0, 0, 0);
		m_e->free_buffer(a);
		m_e->free_buffer(b);
	}
	{
		nbody_engine_memory_fake	b(0);
		nbody_engine::memory*		c = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		m_e->fmadd(a, &b, c, 0, 0, 0, 0);
		m_e->free_buffer(c);
		m_e->free_buffer(a);
	}

	qDebug() << "fmaddn_inplace";
	{
		nbody_engine_memory_fake	a(0);
		nbody_engine::memory*		b = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		c = m_e->create_buffer(m_e->problem_size());
		m_e->fmaddn_inplace(&a, b, c, 0, 0, 0, 0);
		m_e->free_buffer(b);
		m_e->free_buffer(c);
	}
	{
		nbody_engine_memory_fake	c(0);
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		b = m_e->create_buffer(m_e->problem_size());
		m_e->fmaddn_inplace(a, b, &c, 0, 0, 0, 0);
		m_e->free_buffer(a);
		m_e->free_buffer(b);
	}
	{
		nbody_engine_memory_fake	b(0);
		nbody_engine::memory*		c = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		m_e->fmaddn_inplace(a, &b, c, 0, 0, 0, 0);
		m_e->free_buffer(c);
		m_e->free_buffer(a);
	}

	qDebug() << "fmaddn";
	{
		nbody_engine_memory_fake	a(0);
		nbody_engine::memory*		b = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		c = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		d = m_e->create_buffer(m_e->problem_size());
		m_e->fmaddn(&a, b, c, d, 0, 0, 0, 0, 0);
		m_e->fmaddn(&a, NULL, c, d, 0, 0, 0, 0, 0);
		m_e->free_buffer(b);
		m_e->free_buffer(c);
		m_e->free_buffer(d);
	}
	{
		nbody_engine_memory_fake	d(0);
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		b = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		c = m_e->create_buffer(m_e->problem_size());
		m_e->fmaddn(a, b, c, &d, 0, 0, 0, 0, 0);
		m_e->fmaddn(a, NULL, c, &d, 0, 0, 0, 0, 0);
		m_e->free_buffer(a);
		m_e->free_buffer(b);
		m_e->free_buffer(c);
	}
	{
		nbody_engine_memory_fake	c(0);
		nbody_engine::memory*		d = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		b = m_e->create_buffer(m_e->problem_size());
		m_e->fmaddn(a, b, &c, d, 0, 0, 0, 0, 0);
		m_e->fmaddn(a, NULL, &c, d, 0, 0, 0, 0, 0);
		m_e->free_buffer(d);
		m_e->free_buffer(a);
		m_e->free_buffer(b);
	}
	{
		nbody_engine_memory_fake	b(0);
		nbody_engine::memory*		c = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		d = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		m_e->fmaddn(a, &b, c, d, 0, 0, 0, 0, 0);
		m_e->free_buffer(c);
		m_e->free_buffer(d);
		m_e->free_buffer(a);
	}

	qDebug() << "fmaxabs";
	{
		nbody_engine_memory_fake	a(0);
		nbcoord_t					res = 0;
		m_e->fmaxabs(&a, res);
	}
}

int main(int argc, char* argv[])
{
	int res = 0;

	{
		test_nbody_engine tc1(new nbody_engine_block());
		res += QTest::qExec(&tc1, argc, argv);
	}
#ifdef HAVE_OPENCL
	{
		test_nbody_engine tc1(new nbody_engine_opencl());
		res += QTest::qExec(&tc1, argc, argv);
	}
#endif

	{
		test_nbody_engine tc1(new nbody_engine_openmp());
		res += QTest::qExec(&tc1, argc, argv);
	}

	{
		test_nbody_engine tc3(new nbody_engine_simple());
		res += QTest::qExec(&tc3, argc, argv);
	}

	{
		test_nbody_engine tc3(new nbody_engine_simple_bh(), 128, 1e-2);
		res += QTest::qExec(&tc3, argc, argv);
	}

	return res;
}

#include "test_nbody_engine.moc"
