#include <QApplication>
#include <QtTest>
#include <QDebug>
#include <limits>
#include <omp.h>

#include "nbody_engines.h"
#include "nbody_space_heap_func.h"

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
	std::vector<nbcoord_t>	data1;
	std::vector<nbcoord_t>	data2;
	nbody_engine::memory*	mem1 = e->create_buffer(cnt * sizeof(nbcoord_t));
	nbody_engine::memory*	mem2 = e->create_buffer(cnt * sizeof(nbcoord_t));

	data1.resize(cnt);
	data2.resize(cnt);

	for(size_t i = 0; i != data1.size(); ++i)
	{
		data1[i] = static_cast<nbcoord_t>(i);
	}

	e->write_buffer(mem1, data1.data());
	e->copy_buffer(mem2, mem1);
	e->read_buffer(data2.data(), mem2);

	bool	ret = (0 == memcmp(data1.data(), data2.data(), cnt * sizeof(nbcoord_t)));

	e->free_buffer(mem1);
	e->free_buffer(mem2);

	return ret;
}

bool test_fill_buffer(nbody_engine* e)
{
	const size_t			cnt = 33;
	std::vector<nbcoord_t>	data;
	nbody_engine::memory*	mem = e->create_buffer(cnt * sizeof(nbcoord_t));
	nbcoord_t				value(777);

	data.resize(cnt);

	e->fill_buffer(mem, value);
	e->read_buffer(data.data(), mem);

	bool ret = true;
	for(size_t i = 0; i != data.size(); ++i)
	{
		if(fabs(data[i] - value) > std::numeric_limits<nbcoord_t>::min())
		{
			ret = false;
			break;
		};
	}

	e->free_buffer(mem);

	return ret;
}

bool test_fmadd_inplace(nbody_engine* e)
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

bool test_fmadd(nbody_engine* e)
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	const size_t			size = e->problem_size();
	std::vector<nbcoord_t>	a(e->problem_size());
	std::vector<nbcoord_t>	b(e->problem_size());
	std::vector<nbcoord_t>	c(e->problem_size());
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

	//! a[i] = b[i] + c[i]*d
	e->fmadd(mem_a, mem_b, mem_c, d);

	e->read_buffer(a.data(), mem_a);

	bool	ret = true;
	for(size_t i = 0; i != size; ++i)
	{
		if(fabs((b[i] + c[i]*d) - a[i]) > eps)
		{
			ret = false;
		}
	}

	e->free_buffer(mem_a);
	e->free_buffer(mem_b);
	e->free_buffer(mem_c);

	return ret;
}

bool test_fmaddn_inplace(nbody_engine* e, size_t csize)
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	const size_t			size = e->problem_size();
	const size_t			bstride = size;
	std::vector<nbcoord_t>	a(size);
	std::vector<nbcoord_t>	a_res(size);
	std::vector<nbcoord_t>	b(bstride * csize);
	std::vector<nbcoord_t>	c(csize);
	nbody_engine::memory*		mem_a = e->create_buffer(a.size() * sizeof(nbcoord_t));
	nbody_engine::memory_array	mem_b = e->create_buffers(bstride * sizeof(nbcoord_t), csize);

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
	if(csize > 1)
	{
		c[0] = 0_f;
		c[csize / 2] = 0_f;
	}

	e->write_buffer(mem_a, a.data());
	for(size_t n = 0; n != mem_b.size(); ++n)
	{
		e->write_buffer(mem_b[n], b.data() + n * bstride);
	}

	//! a[i] += sum( b[k][i]*c[k], k=[0...b.size()) )
	e->fmaddn_inplace(mem_a, mem_b, c.data(), c.size());

	e->read_buffer(a_res.data(), mem_a);

	bool	ret = true;
	for(size_t i = 0; i != size; ++i)
	{
		nbcoord_t	s = 0;
		for(size_t k = 0; k != csize; ++k)
		{
			s += b[i + k * bstride] * c[k];
		}
		if(fabs((a[i] + s) - a_res[i]) > eps)
		{
			ret = false;
		}
	}

	e->free_buffer(mem_a);
	e->free_buffers(mem_b);

	return ret;
}

bool test_fmaddn(nbody_engine* e, size_t dsize)
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	const size_t			size = e->problem_size();
	const size_t			cstride = size;
	std::vector<nbcoord_t>	a(e->problem_size());
	std::vector<nbcoord_t>	b(e->problem_size());
	std::vector<nbcoord_t>	c(cstride * dsize);
	std::vector<nbcoord_t>	d(dsize);
	nbody_engine::memory*		mem_a = e->create_buffer(a.size() * sizeof(nbcoord_t));
	nbody_engine::memory*		mem_b = e->create_buffer(b.size() * sizeof(nbcoord_t));
	nbody_engine::memory_array	mem_c = e->create_buffers(cstride * sizeof(nbcoord_t), dsize);

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
	if(dsize > 1)
	{
		d[0] = 0_f;
		d[dsize / 2] = 0_f;
	}

	e->write_buffer(mem_a, a.data());
	e->write_buffer(mem_b, b.data());
	for(size_t n = 0; n != mem_c.size(); ++n)
	{
		e->write_buffer(mem_c[n], c.data() + n * cstride);
	}

	//! a[i] = b[i] + sum( c[k][i]*d[k], k=[0...c.size()) )
	e->fmaddn(mem_a, mem_b, mem_c, d.data(), dsize);

	e->read_buffer(a.data(), mem_a);

	bool	ret = true;
	for(size_t i = 0; i != size; ++i)
	{
		nbcoord_t	s = 0;
		for(size_t k = 0; k != dsize; ++k)
		{
			s += c[i + k * cstride] * d[k];
		}
		if(fabs((b[i] + s) - a[i]) > eps)
		{
			ret = false;
		}
	}

	e->free_buffer(mem_a);
	e->free_buffer(mem_b);
	e->free_buffers(mem_c);

	return ret;
}

bool test_fmaddn_null_b(nbody_engine* e, size_t dsize)
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	const size_t			size = e->problem_size();
	const size_t			cstride = size;
	std::vector<nbcoord_t>	a(e->problem_size());
	std::vector<nbcoord_t>	c(cstride * dsize);
	std::vector<nbcoord_t>	d(dsize);
	nbody_engine::memory*		mem_a = e->create_buffer(a.size() * sizeof(nbcoord_t));
	nbody_engine::memory_array	mem_c = e->create_buffers(cstride * sizeof(nbcoord_t), dsize);

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
	if(dsize > 1)
	{
		d[0] = 0_f;
		d[dsize / 2] = 0_f;
	}

	e->write_buffer(mem_a, a.data());
	for(size_t n = 0; n != mem_c.size(); ++n)
	{
		e->write_buffer(mem_c[n], c.data() + n * cstride);
	}

	//! a[i] = b[i] + sum( c[k][i]*d[k], k=[0...c.size()) )
	e->fmaddn(mem_a, NULL, mem_c, d.data(), dsize);

	e->read_buffer(a.data(), mem_a);

	bool	ret = true;
	for(size_t i = 0; i != size; ++i)
	{
		nbcoord_t	s = 0;
		for(size_t k = 0; k < dsize; ++k)
		{
			s += c[i + k * cstride] * d[k];
		}
		if(fabs(s - a[i]) > eps)
		{
			ret = false;
		}
	}

	e->free_buffer(mem_a);
	e->free_buffers(mem_c);

	return ret;
}

bool test_fmaddn_corr(nbody_engine* e, size_t csize)
{
	nbcoord_t				eps = std::numeric_limits<nbcoord_t>::epsilon();
	const size_t			size = e->problem_size();
	const size_t			bstride = size;
	std::vector<nbcoord_t>	a(size);
	std::vector<nbcoord_t>	corr(size);
	std::vector<nbcoord_t>	a_res(size);
	std::vector<nbcoord_t>	b(bstride * csize);
	std::vector<nbcoord_t>	c(csize);
	nbody_engine::memory*		mem_a = e->create_buffer(a.size() * sizeof(nbcoord_t));
	nbody_engine::memory*		mem_corr = e->create_buffer(corr.size() * sizeof(nbcoord_t));
	nbody_engine::memory_array	mem_b = e->create_buffers(bstride * sizeof(nbcoord_t), csize);

	for(size_t n = 0; n != a.size(); ++n)
	{
		a[n] = 1_f;
	}
	for(size_t n = 0; n != b.size(); ++n)
	{
		b[n] = 1_f;
	}
	nbcoord_t	factor = eps / 2_f;
	for(size_t n = 0; n != c.size(); ++n)
	{
		c[n] = factor;
	}

	e->write_buffer(mem_a, a.data());
	e->fill_buffer(mem_corr, 0_f);
	for(size_t n = 0; n != mem_b.size(); ++n)
	{
		e->write_buffer(mem_b[n], b.data() + n * bstride);
	}

	//! a[i] += sum( b[k][i]*c[k], k=[0...b.size()) )
	e->fmaddn_corr(mem_a, mem_corr, mem_b, c.data(), c.size());

	e->read_buffer(a_res.data(), mem_a);

	bool	ret = true;
	for(size_t i = 0; i != size; ++i)
	{
		nbcoord_t	s = csize * factor;
		nbcoord_t	absdelta = fabs((a[i] + s) - a_res[i]);
		if(absdelta > eps)
		{
			if(ret)
			{
				qDebug() << "Invalid delta" << absdelta << "at index" << i;
			}
			ret = false;
		}
	}

	e->free_buffer(mem_corr);
	e->free_buffer(mem_a);
	e->free_buffers(mem_b);

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

	bool	ret = (fabs(result - testmax) <= eps);

	e->free_buffer(mem_a);

	return ret;
}

bool test_fcompute(nbody_engine* e0, nbody_engine* e1, nbody_data* data,
				   const nbcoord_t eps, size_t step_count)
{
	nbody_engine::memory*	e0y = e0->create_buffer(e0->get_y()->size());
	nbody_engine::memory*	e1y = e1->create_buffer(e1->get_y()->size());

	e0->copy_buffer(e0y, e0->get_y());
	e1->copy_buffer(e1y, e1->get_y());

	for(size_t step = 0; step != step_count; ++step)
	{
		std::vector<nbcoord_t>	f0(e0->problem_size());

		{
			double					tbegin = omp_get_wtime();
			nbody_engine::memory*	fbuff;
			fbuff = e0->create_buffer(sizeof(nbcoord_t) * e0->problem_size());
			e0->fill_buffer(fbuff, 1e10);
			e0->set_step(step);
			e0->fcompute(0, e0y, fbuff);

			e0->read_buffer(f0.data(), fbuff);
			e0->free_buffer(fbuff);
			qDebug() << "Time" << e0->type_name() << omp_get_wtime() - tbegin;
		}

		std::vector<nbcoord_t>	f(e1->problem_size());

		{
			double					tbegin = omp_get_wtime();
			nbody_engine::memory*	fbuff;
			fbuff = e1->create_buffer(sizeof(nbcoord_t) * e1->problem_size());
			e1->fill_buffer(fbuff, -1e10);
			e1->set_step(step);
			e1->fcompute(0, e1y, fbuff);

			e1->read_buffer(f.data(), fbuff);
			e1->free_buffer(fbuff);
			qDebug() << "Time" << e1->type_name() << omp_get_wtime() - tbegin;
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
			qDebug() << "Stars count:         " << data->get_count();
			qDebug() << "Problem size:        " << e1->problem_size();
			qDebug() << "Step:                " << step + 1 << "from" << step_count;
			qDebug() << "Total count:         " << f.size();
			qDebug() << "Total error:         " << total_err;
			qDebug() << "Mean error:          " << total_err / f.size();
			qDebug() << "Total relative error:" << total_relative_err;
			qDebug() << "Outliers count:      " << outliers_count;
			e0->free_buffer(e0y);
			e1->free_buffer(e1y);
			return false;
		}
		e0->fmadd_inplace(e0y, e0y, 0.99);
		e1->fmadd_inplace(e1y, e1y, 0.99);
	}
	e0->free_buffer(e0y);
	e1->free_buffer(e1y);
	return true;
}

bool test_fcompute(nbody_engine* e, nbody_data* data, const nbcoord_t eps)
{
	nbody_engine_simple		e0;
	e0.init(data);

	return test_fcompute(&e0, e, data, eps, 1);
}

bool test_clamp(nbody_engine* e)
{
	std::vector<nbcoord_t>	a(e->problem_size());
	std::vector<nbcoord_t>	b(e->problem_size());
	nbody_engine::memory*	mem_a = e->create_buffer(sizeof(nbcoord_t) * a.size());

	for(size_t n = 0; n != a.size(); ++n)
	{
		if(n & 1)
		{
			a[n] = +4;
		}
		else
		{
			a[n] = -4;
		}
	}
	e->write_buffer(mem_a, a.data());
	e->clamp(mem_a, 1);
	e->read_buffer(b.data(), mem_a);

	bool ok = true;
	for(size_t n = 0; n != b.size(); ++n)
	{
		bool eq = true;
		if(n < b.size() / 2)
		{
			if(n & 1)
			{
				eq = (b[n] == +2);
			}
			else
			{
				eq = (b[n] == -2);
			}
		}
		else
		{
			if(n & 1)
			{
				eq = (b[n] == +4);
			}
			else
			{
				eq = (b[n] == -4);
			}
		}
		if(ok && !eq)
		{
			qDebug() << "test_clamp failed at index" << n << "a[n]" << a[n] << "b[n]" << b[n];
		}
		ok = ok && eq;
	}

	e->free_buffer(mem_a);

	return ok;
}

class test_nbody_engine : public QObject
{
	Q_OBJECT

	nbody_data		m_data;
	nbody_engine*	m_e;
	size_t			m_problem_size;
	nbcoord_t		m_eps;
public:
	explicit test_nbody_engine(nbody_engine* e, size_t problen_size = 64, nbcoord_t eps = 1e-13);
	~test_nbody_engine();

private slots:
	void initTestCase();
	void cleanupTestCase();
	void test_mem();
	void test_memcpy();
	void test_copy_buffer();
	void test_fill_buffer();
	void test_fmadd1();
	void test_fmadd();
	void test_fmaddn_inplace();
	void test_fmaddn();
	void test_fmaddn_null_b();
	void test_fmaddn_corr();
	void test_fmaxabs();
	void test_fcompute();
	void test_clamp();
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
	QVERIFY(m_e != nullptr);
	nbcoord_t				box_size = 100;

	qDebug() << "Engine" << m_e->type_name() << "Problem size" << m_problem_size;
	m_e->print_info();
	m_data.make_universe(m_problem_size, box_size, box_size, box_size);
	QVERIFY(m_e->init(&m_data));
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

void test_nbody_engine::test_fill_buffer()
{
	QVERIFY(::test_fill_buffer(m_e));
}

void test_nbody_engine::test_fmadd1()
{
	QVERIFY(::test_fmadd_inplace(m_e));
}

void test_nbody_engine::test_fmadd()
{
	QVERIFY(::test_fmadd(m_e));
}

void test_nbody_engine::test_fmaddn_inplace()
{
	QVERIFY(::test_fmaddn_inplace(m_e, 1));
	QVERIFY(::test_fmaddn_inplace(m_e, 3));
	QVERIFY(::test_fmaddn_inplace(m_e, 7));
}

void test_nbody_engine::test_fmaddn()
{
	QVERIFY(::test_fmaddn(m_e, 1));
	QVERIFY(::test_fmaddn(m_e, 3));
	QVERIFY(::test_fmaddn(m_e, 7));
}

void test_nbody_engine::test_fmaddn_null_b()
{
	QVERIFY(::test_fmaddn_null_b(m_e, 1));
	QVERIFY(::test_fmaddn_null_b(m_e, 3));
	QVERIFY(::test_fmaddn_null_b(m_e, 7));
}

void test_nbody_engine::test_fmaddn_corr()
{
	QVERIFY(::test_fmaddn_corr(m_e, 10));
}

void test_nbody_engine::test_fmaxabs()
{
	QVERIFY(::test_fmaxabs(m_e));
}

void test_nbody_engine::test_fcompute()
{
	QVERIFY(::test_fcompute(m_e, &m_data, m_eps));
	m_data.advise_time(0);
	QVERIFY(::test_fcompute(m_e, &m_data, m_eps));
}

void test_nbody_engine::test_clamp()
{
	QVERIFY(::test_clamp(m_e));
}

class nbody_engine_memory_fake : public nbody_engine::memory
{
	size_t m_size;
public:
	explicit nbody_engine_memory_fake(size_t sz) : m_size(sz)
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
		m_e->fcompute(0, &y, f);
		m_e->free_buffer(f);
	}
	{
		nbody_engine_memory_fake	f(0);
		nbody_engine::memory*		y = m_e->create_buffer(m_e->problem_size());
		m_e->fcompute(0, y, &f);
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
		m_e->copy_buffer(&a, b);
		m_e->free_buffer(b);
	}
	{
		nbody_engine_memory_fake	b(0);
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		m_e->copy_buffer(a, &b);
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
		m_e->fmadd(&a, b, c, 0);
		m_e->free_buffer(b);
		m_e->free_buffer(c);
	}
	{
		nbody_engine_memory_fake	c(0);
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		b = m_e->create_buffer(m_e->problem_size());
		m_e->fmadd(a, b, &c, 0);
		m_e->free_buffer(a);
		m_e->free_buffer(b);
	}
	{
		nbody_engine_memory_fake	b(0);
		nbody_engine::memory*		c = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		m_e->fmadd(a, &b, c, 0);
		m_e->free_buffer(c);
		m_e->free_buffer(a);
	}

	qDebug() << "fmaddn_inplace";
	{
		nbody_engine_memory_fake	a(0);
		nbody_engine::memory_array	b = m_e->create_buffers(m_e->problem_size(), 1);
		nbcoord_t					c[1] = {};
		m_e->fmaddn_inplace(&a, b, c, 1);
		m_e->free_buffers(b);
	}
	{
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory_array	b = m_e->create_buffers(m_e->problem_size(), 1);
		m_e->fmaddn_inplace(a, b, NULL, 1);
		m_e->free_buffer(a);
		m_e->free_buffers(b);
	}
	{
		nbody_engine_memory_fake	b0(0);
		nbody_engine::memory_array	b(1, &b0);
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		nbcoord_t					c[1] = {};
		m_e->fmaddn_inplace(a, b, c, 1);
		m_e->free_buffer(a);
	}
	{
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory_array	b = m_e->create_buffers(m_e->problem_size(), 1);
		nbcoord_t					c[1] = {};
		m_e->fmaddn_inplace(a, b, c, 100);
		m_e->free_buffer(a);
		m_e->free_buffers(b);
	}
	{
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory_array	b = m_e->create_buffers(m_e->problem_size(), 1);
		nbody_engine::memory_array	badb(b);
		nbcoord_t					c[1] = {1};
		badb[0] = nullptr;
		m_e->fmaddn_inplace(a, badb, c, 1);
		m_e->free_buffer(a);
		m_e->free_buffers(b);
	}
	qDebug() << "fmaddn_corr";
	{
		nbody_engine::memory*		a = nullptr;
		nbody_engine::memory*		corr = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory_array	b = m_e->create_buffers(m_e->problem_size(), 1);
		nbcoord_t					c[1] = {};
		m_e->fmaddn_corr(a, corr, b, c, 1);
		m_e->free_buffer(corr);
		m_e->free_buffers(b);
	}
	{
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		corr = nullptr;
		nbody_engine::memory_array	b = m_e->create_buffers(m_e->problem_size(), 1);
		nbcoord_t					c[1] = {};
		m_e->fmaddn_corr(a, corr, b, c, 1);
		m_e->free_buffer(a);
		m_e->free_buffers(b);
	}
	{
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		corr = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory_array	b(1, nullptr);
		nbcoord_t					c[1] = {};
		m_e->fmaddn_corr(a, corr, b, c, 1);
		m_e->free_buffer(a);
		m_e->free_buffer(corr);
	}
	{
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		corr = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory_array	b = m_e->create_buffers(m_e->problem_size(), 1);
		m_e->fmaddn_corr(a, corr, b, nullptr, 1);
		m_e->free_buffer(a);
		m_e->free_buffer(corr);
		m_e->free_buffers(b);
	}
	{
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		corr = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory_array	b = m_e->create_buffers(m_e->problem_size(), 1);
		nbcoord_t					c[1] = {};
		m_e->fmaddn_corr(a, corr, b, c, 100);
		m_e->free_buffer(a);
		m_e->free_buffer(corr);
		m_e->free_buffers(b);
	}
	{
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		corr = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory_array	b = m_e->create_buffers(m_e->problem_size(), 1);
		nbody_engine::memory_array	badb(b);
		nbcoord_t					c[1] = {1};
		badb[0] = nullptr;
		m_e->fmaddn_corr(a, corr, badb, c, 1);
		m_e->free_buffer(a);
		m_e->free_buffer(corr);
		m_e->free_buffers(b);
	}
	qDebug() << "fmaddn";
	{
		nbody_engine_memory_fake	a(0);
		nbody_engine::memory*		b = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory_array	c = m_e->create_buffers(m_e->problem_size(), 1);
		nbcoord_t					d[1] = {};
		m_e->fmaddn(&a, b, c, d, 200000);
		m_e->free_buffer(b);
		m_e->free_buffers(c);
	}
	{
		nbody_engine_memory_fake	a(0);
		nbody_engine::memory*		b = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory_array	c = m_e->create_buffers(m_e->problem_size(), 1);
		nbcoord_t					d[1] = {};
		m_e->fmaddn(&a, b, c, d, 1);
		m_e->fmaddn(&a, NULL, c, d, 1);
		m_e->free_buffer(b);
		m_e->free_buffers(c);
	}
	{
		nbcoord_t*					d(NULL);
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		b = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory_array	c = m_e->create_buffers(m_e->problem_size(), 1);
		m_e->fmaddn(a, b, c, d, 1);
		m_e->fmaddn(a, NULL, c, d, 1);
		m_e->free_buffer(a);
		m_e->free_buffer(b);
		m_e->free_buffers(c);
	}
	{
		nbody_engine_memory_fake	c0(0);
		nbody_engine::memory_array	c(1, &c0);
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		b = m_e->create_buffer(m_e->problem_size());
		nbcoord_t					d[1] = {};
		m_e->fmaddn(a, b, c, d, 1);
		m_e->fmaddn(a, NULL, c, d, 1);
		m_e->free_buffer(a);
		m_e->free_buffer(b);
	}
	{
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		nbody_engine_memory_fake	b(0);
		nbody_engine::memory_array	c = m_e->create_buffers(m_e->problem_size(), 1);
		nbcoord_t					d[1] = {};
		m_e->fmaddn(a, &b, c, d, 1);
		m_e->free_buffer(a);
		m_e->free_buffers(c);
	}
	{
		nbody_engine::memory*		a = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory*		b = m_e->create_buffer(m_e->problem_size());
		nbody_engine::memory_array	c = m_e->create_buffers(m_e->problem_size(), 1);
		nbody_engine::memory_array	badc(c);
		nbcoord_t					d[1] = {1};
		badc[0] = nullptr;
		m_e->fmaddn(a, b, badc, d, 1);
		m_e->free_buffer(a);
		m_e->free_buffer(b);
		m_e->free_buffers(c);
	}
	qDebug() << "fmaxabs";
	{
		nbody_engine_memory_fake	a(0);
		nbcoord_t					res = 0;
		m_e->fmaxabs(&a, res);
	}
}

class test_nbody_engine_compare : public QObject
{
	Q_OBJECT
	nbody_data		m_data;
	nbody_engine*	m_e1;
	nbody_engine*	m_e2;
	size_t			m_problem_size;
	nbcoord_t		m_eps;
	size_t			m_step_count;
public:
	test_nbody_engine_compare(nbody_engine* e1, nbody_engine* e2,
							  size_t problen_size = 64,
							  nbcoord_t eps = 1e-13,
							  size_t step_count = 1);
	~test_nbody_engine_compare();
private slots:
	void initTestCase();
	void compare();
};

test_nbody_engine_compare::test_nbody_engine_compare(nbody_engine* e1, nbody_engine* e2,
													 size_t problen_size, nbcoord_t eps,
													 size_t step_count) :
	m_e1(e1),
	m_e2(e2),
	m_problem_size(problen_size),
	m_eps(eps),
	m_step_count(step_count)
{
}

test_nbody_engine_compare::~test_nbody_engine_compare()
{
	delete m_e1;
	delete m_e2;
}

void test_nbody_engine_compare::initTestCase()
{
	QVERIFY(m_e1 != nullptr);
	QVERIFY(m_e2 != nullptr);
	nbcoord_t				box_size = 100;

	qDebug() << "Problem size" << m_problem_size;
	qDebug() << "Engine1" << m_e1->type_name();
	m_e1->print_info();
	qDebug() << "Engine2" << m_e2->type_name();
	m_e2->print_info();
	m_data.make_universe(m_problem_size, box_size, box_size, box_size);
	QVERIFY(m_e1->init(&m_data));
	QVERIFY(m_e2->init(&m_data));
}

void test_nbody_engine_compare::compare()
{
	QVERIFY(::test_fcompute(m_e1, m_e2, &m_data, m_eps, m_step_count));
}

class test_nbody_heap_func : public QObject
{
	Q_OBJECT
	typedef nbody_heap_func<size_t>	hf;
	static const size_t	s_tree_size = 16;
	static size_t __ffs(size_t x)
	{
		if(x == 0)
		{
			return 0;
		}
		size_t t = 1;
		size_t r = 1;
		while((x & t) == 0)
		{
			t = t << 1;
			r++;
		}
		return r;
	}
	static size_t cu_next_down(size_t idx)
	{
		idx = (idx >> (__ffs(~idx) - 1));
		return hf::left2right(idx);
	}

public:
	test_nbody_heap_func() {}
	~test_nbody_heap_func() {}

	// Heap-tree example:
	// 8  9   10  11  12  13  14  15
	//   4       5       6       7
	//       2               3
	//               1
private slots:
	void next_down()
	{
		for(size_t idx = NBODY_HEAP_ROOT_INDEX; idx != s_tree_size; ++idx)
		{
			QCOMPARE(cu_next_down(idx), hf::next_down(idx));
		}
		QCOMPARE(hf::next_down(8), static_cast<size_t>(9));
		QCOMPARE(hf::next_down(9), static_cast<size_t>(5));
	}
	void skip_idx()
	{
		QCOMPARE(hf::skip_idx(8), static_cast<size_t>(9));
		QCOMPARE(hf::skip_idx(9), static_cast<size_t>(5));
		QCOMPARE(hf::skip_idx(5), static_cast<size_t>(3));
		QCOMPARE(hf::skip_idx(11), static_cast<size_t>(3));
		QCOMPARE(hf::skip_idx(3), static_cast<size_t>(1));
		QCOMPARE(hf::skip_idx(7), static_cast<size_t>(1));
		QCOMPARE(hf::skip_idx(15), static_cast<size_t>(1));
	}
	void next_up()
	{
		QCOMPARE(hf::next_up(1, s_tree_size), static_cast<size_t>(2));
		QCOMPARE(hf::next_up(8, s_tree_size), static_cast<size_t>(9));
		QCOMPARE(hf::next_up(9, s_tree_size), static_cast<size_t>(5));
		QCOMPARE(hf::next_up(15, s_tree_size), static_cast<size_t>(1));
	}
	void parent_idx()
	{
		for(size_t idx = NBODY_HEAP_ROOT_INDEX; idx != s_tree_size; ++idx)
		{
			QCOMPARE(hf::parent_idx(hf::left_idx(idx)), idx);
			QCOMPARE(hf::parent_idx(hf::rght_idx(idx)), idx);
		}
	}
	void left2right()
	{
		for(size_t idx = NBODY_HEAP_ROOT_INDEX; idx != s_tree_size; ++idx)
		{
			QCOMPARE(hf::left2right(hf::left_idx(idx)), hf::rght_idx(idx));
		}
	}
};

static int common_test(int argc, char** argv)
{
	int res = 0;
	QVariantMap param0(std::map<QString, QVariant>({{"engine", "invalid"}}));
	nbody_engine*	e0(nbody_create_engine(param0));
	if(e0 != NULL)
	{
		qDebug() << "Created engine with invalid type" << param0;
		res += 1;
		delete e0;
	}
	{
		test_nbody_heap_func	tc1;
		res += QTest::qExec(&tc1, argc, argv);
	}
	return res;
}

static int ah_test(int argc, char** argv)
{
	int res = 0;
	QVariantMap			param(std::map<QString, QVariant>({{"engine", "ah"},
		{"full_recompute_rate", 5}, {"max_dist", 7}, {"min_force", 1e-6}
	}));
	test_nbody_engine	tc1(nbody_create_engine(param));
	res += QTest::qExec(&tc1, argc, argv);
	return res;
}

static int block_test(int argc, char** argv)
{
	int res = 0;
	QVariantMap			param(std::map<QString, QVariant>({{"engine", "block"}}));
	test_nbody_engine	tc1(nbody_create_engine(param));
	res += QTest::qExec(&tc1, argc, argv);
	return res;
}

#ifdef HAVE_CUDA
static int cuda_test(int argc, char** argv)
{
	int res = 0;
#ifdef HAVE_NCCL
	{
		QVariantMap param1(std::map<QString, QVariant>({{"engine", "cuda"},
			{"device", "0"}
		}));
		QVariantMap param2(std::map<QString, QVariant>({{"engine", "cuda"},
			{"device", "0,1"},
			{"use_nccl", true}
		}));
		test_nbody_engine	tc1(nbody_create_engine(param2), 128);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param1(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"tree_build_rate", 0},
			{"traverse_type", "nested_tree"},
			{"tree_layout", "heap"}
		}));
		QVariantMap param2(std::map<QString, QVariant>({{"engine", "cuda_bh"},
			{"block_size", 16},
			{"distance_to_node_radius_ratio", 3.1623},
			{"tree_build_rate", 0},
			{"traverse_type", "nested_tree"},
			{"tree_layout", "heap"},
			{"device", "0,1"},
			{"use_nccl", true}
		}));
		test_nbody_engine_compare tc1(nbody_create_engine(param1),
									  nbody_create_engine(param2),
									  128, 1e-13, 2);
		res += QTest::qExec(&tc1, argc, argv);
	}
	return res;
#endif// HAVE_NCCL
	for(const char* engine : {"cuda", "cuda_bh", "cuda_bh_tex"})
	{
		for(const char* device : {"", "a", "0,a", "-1", "9999"})
		{
			QVariantMap		params(std::map<QString, QVariant>({{"engine", engine}, {"device", device}}));
			nbody_engine*	e(nbody_create_engine(params));
			if(e != NULL)
			{
				qDebug() << "Created engine with invalid device" << params;
				res += 1;
				delete e;
			}
		}
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"engine", "cuda"},
			{"block_size", 64}
		}));
		test_nbody_engine	tc1(nbody_create_engine(param));
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"engine", "cuda"},
			{"block_size", 128}
		}));
		test_nbody_engine	tc1(nbody_create_engine(param), 128);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param1(std::map<QString, QVariant>({{"engine", "cuda"},
			{"device", "0"}
		}));
		QVariantMap param2(std::map<QString, QVariant>({{"engine", "cuda"},
			{"device", "0,0"}
		}));
		test_nbody_engine	tc1(nbody_create_engine(param2), 128);
		res += QTest::qExec(&tc1, argc, argv);
		test_nbody_engine_compare tc2(nbody_create_engine(param1),
									  nbody_create_engine(param2),
									  1024, 1e-15);
		res += QTest::qExec(&tc2, argc, argv);
	}
	for(const int tree_build_rate : {0, 2})
	{
		for(const char* device : {"0", "0,0"})
		{
			QVariantMap param1(std::map<QString, QVariant>({{"engine", "simple_bh"},
				{"distance_to_node_radius_ratio", 3.1623},
				{"tree_build_rate", tree_build_rate},
				{"traverse_type", "nested_tree"},
				{"tree_layout", "heap"}
			}));
			QVariantMap param2(std::map<QString, QVariant>({{"engine", "cuda_bh"},
				{"block_size", 16},
				{"distance_to_node_radius_ratio", 3.1623},
				{"tree_build_rate", tree_build_rate},
				{"traverse_type", "nested_tree"},
				{"tree_layout", "heap"},
				{"device", device}
			}));
			test_nbody_engine_compare tc1(nbody_create_engine(param1),
										  nbody_create_engine(param2),
										  128, 1e-13, 2);
			res += QTest::qExec(&tc1, argc, argv);
		}
	}
	for(const int tree_build_rate : {0, 2})
	{
		for(const char* device : {"0", "0,0", "0,0,0,0"})
		{
			for(const char* tree_layout : {"heap", "heap_stackless"})
			{
				for(const double distance_to_node_radius_ratio : {3.1623, 1e8})
				{
					QVariantMap param1(std::map<QString, QVariant>({{"engine", "simple_bh"},
						{"distance_to_node_radius_ratio", distance_to_node_radius_ratio},
						{"tree_build_rate", tree_build_rate},
						{"traverse_type", "nested_tree"},
						{"tree_layout", "heap"}
					}));
					QVariantMap param2(std::map<QString, QVariant>({{"engine", "cuda_bh_tex"},
						{"distance_to_node_radius_ratio", distance_to_node_radius_ratio},
						{"tree_build_rate", tree_build_rate},
						{"traverse_type", "nested_tree"},
						{"tree_layout", tree_layout},
						{"device", device}
					}));
					test_nbody_engine_compare tc1(nbody_create_engine(param1),
												  nbody_create_engine(param2),
												  128, 1e-13, 2);
					res += QTest::qExec(&tc1, argc, argv);
				}
			}
		}
	}
	{
		QVariantMap param2(std::map<QString, QVariant>({{"engine", "cuda_bh_tex"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "nested_tree"},
			{"tree_layout", "tree"}
		}));
		nbody_engine*	e2(nbody_create_engine(param2));
		if(e2 != NULL)
		{
			qDebug() << "Created engine with invalid tree_layout" << param2;
			res += 1;
			delete e2;
		}
	}
	return res;
}
#endif //HAVE_CUDA

static int openmp_test(int argc, char** argv)
{
	int res = 0;
	QVariantMap			param(std::map<QString, QVariant>({{"engine", "openmp"}}));
	test_nbody_engine	tc1(nbody_create_engine(param));
	res += QTest::qExec(&tc1, argc, argv);
	return res;
}

static int simple_test(int argc, char** argv)
{
	int res = 0;
	QVariantMap			param(std::map<QString, QVariant>({{"engine", "simple"}}));
	test_nbody_engine	tc1(nbody_create_engine(param));
	res += QTest::qExec(&tc1, argc, argv);
	return res;
}

static int simple_bh_test(int argc, char** argv)
{
	int res = 0;
	{
		QVariantMap param(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 1e8},
			{"traverse_type", "cycle"},
			{"tree_layout", "tree"}
		}));
		test_nbody_engine tc1(nbody_create_engine(param), 128, 1e-11);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 1e8},
			{"traverse_type", "cycle"},
			{"tree_layout", "heap"}
		}));
		test_nbody_engine tc1(nbody_create_engine(param), 128, 1e-11);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 1e8},
			{"traverse_type", "nested_tree"},
			{"tree_layout", "tree"}
		}));
		test_nbody_engine tc1(nbody_create_engine(param), 128, 1e-11);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 1e8},
			{"traverse_type", "nested_tree"},
			{"tree_layout", "heap"}
		}));
		test_nbody_engine tc1(nbody_create_engine(param), 128, 1e-11);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "cycle"},
			{"tree_layout", "tree"}
		}));
		test_nbody_engine tc1(nbody_create_engine(param), 128, 1e-2);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "cycle"},
			{"tree_layout", "heap"}
		}));
		test_nbody_engine tc1(nbody_create_engine(param), 128, 1e-2);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param1(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "cycle"},
			{"tree_layout", "tree"}
		}));
		QVariantMap param2(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "cycle"},
			{"tree_layout", "heap"}
		}));
		test_nbody_engine_compare tc1(nbody_create_engine(param1),
									  nbody_create_engine(param2),
									  1024, 1e-14);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param1(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "cycle"},
			{"tree_layout", "tree"}
		}));
		QVariantMap param2(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "nested_tree"},
			{"tree_layout", "tree"}
		}));
		test_nbody_engine_compare tc1(nbody_create_engine(param1),
									  nbody_create_engine(param2),
									  1024, 1e-14);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param1(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "cycle"},
			{"tree_layout", "heap"}
		}));
		QVariantMap param2(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "nested_tree"},
			{"tree_layout", "heap"}
		}));
		test_nbody_engine_compare tc1(nbody_create_engine(param1),
									  nbody_create_engine(param2),
									  1024, 1e-14);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param1(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 1e8},
			{"traverse_type", "nested_tree"},
			{"tree_layout", "heap"}
		}));
		QVariantMap param2(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 1e8},
			{"traverse_type", "nested_tree"},
			{"tree_layout", "heap_stackless"}
		}));
		test_nbody_engine_compare tc1(nbody_create_engine(param1),
									  nbody_create_engine(param2),
									  1024, 1e-16);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param1(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "nested_tree"},
			{"tree_layout", "heap"}
		}));
		QVariantMap param2(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "nested_tree"},
			{"tree_layout", "heap_stackless"}
		}));
		test_nbody_engine_compare tc1(nbody_create_engine(param1),
									  nbody_create_engine(param2),
									  1024, 1e-16);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param1(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 1e8},
			{"traverse_type", "cycle"},
			{"tree_layout", "tree"},
			{"tree_build_rate", 0}
		}));
		QVariantMap param2(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 1e8},
			{"traverse_type", "cycle"},
			{"tree_layout", "tree"},
			{"tree_build_rate", 2}
		}));
		test_nbody_engine_compare tc1(nbody_create_engine(param1),
									  nbody_create_engine(param2),
									  1024, 1e-16, 2);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param1(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 1e8},
			{"traverse_type", "cycle"},
			{"tree_layout", "heap"},
			{"tree_build_rate", 0}
		}));
		QVariantMap param2(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 1e8},
			{"traverse_type", "cycle"},
			{"tree_layout", "heap"},
			{"tree_build_rate", 2}
		}));
		test_nbody_engine_compare tc1(nbody_create_engine(param1),
									  nbody_create_engine(param2),
									  1024, 1e-16, 2);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param1(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "cycle"},
			{"tree_layout", "invalid"}
		}));
		QVariantMap param2(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "invalid"},
			{"tree_layout", "heap"}
		}));
		nbody_engine*	e1(nbody_create_engine(param1));
		if(e1 != NULL)
		{
			qDebug() << "Created engine with invalid tree_layout" << param1;
			res += 1;
			delete e1;
		}
		nbody_engine*	e2(nbody_create_engine(param2));
		if(e2 != NULL)
		{
			qDebug() << "Created engine with invalid traverse_type" << param2;
			res += 1;
			delete e2;
		}
	}
	return res;
}

#ifdef HAVE_OPENCL
static int opencl_test(int argc, char** argv)
{
	int res = 0;
	{
		QVariantMap	param1(std::map<QString, QVariant>({{"engine", "opencl"}, {"device", "77:0"}}));
		QVariantMap	param2(std::map<QString, QVariant>({{"engine", "opencl"}, {"device", "0:77"}}));
		QVariantMap	param3(std::map<QString, QVariant>({{"engine", "opencl"}, {"device", "0:0:0"}}));
		QVariantMap	param4(std::map<QString, QVariant>({{"engine", "opencl"}, {"device", "0"}}));
		QVariantMap	param5(std::map<QString, QVariant>({{"engine", "opencl"}, {"device", "a:0"}}));
		QVariantMap	param6(std::map<QString, QVariant>({{"engine", "opencl"}, {"device", "0:a"}}));
		QVariantMap	param7(std::map<QString, QVariant>({{"engine", "opencl"}, {"device", "0:0,0,0"}}));
		QVariantMap	param8(std::map<QString, QVariant>({{"engine", "opencl_bh"}, {"traverse_type", "infinite"}}));
		QVariantMap	param9(std::map<QString, QVariant>({{"engine", "opencl_bh"}, {"tree_layout", "garbage"}}));
		QVariantMap	params[] = {param1, param2, param3, param4, param5, param6, param7, param8, param9};
		for(const auto& param : params)
		{
			nbody_engine*	e(nbody_create_engine(param));
			if(e != NULL)
			{
				qDebug() << "Created engine with invalid device" << param;
				res += 1;
				delete e;
			}
		}
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"engine", "opencl"},
			{"verbose", "1"},
			{"oclprof", "1"},
			{"block_size", 32}
		}));
		test_nbody_engine	tc1(nbody_create_engine(param));
		res += QTest::qExec(&tc1, argc, argv);
	}
	for(const char* device : {"0:0;0:0", "0:0,0", "0:0,0;0:0,0", "0:0,0,0,0", "0:0,0,0,0,0,0,0,0"})
	{
		QVariantMap			param1(std::map<QString, QVariant>({{"engine", "opencl"}, {"block_size", 16}, {"device", "0:0"}}));
		QVariantMap			param2(param1); param2["device"] = device;
		test_nbody_engine	tc1(nbody_create_engine(param2));
		res += QTest::qExec(&tc1, argc, argv);
		test_nbody_engine_compare tc2(nbody_create_engine(param1),
									  nbody_create_engine(param2),
									  64, 1e-16, 2);
		res += QTest::qExec(&tc2, argc, argv);
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"engine", "opencl_bh"},
			{"verbose", "1"},
			{"distance_to_node_radius_ratio", 1e8},
			{"oclprof", "1"}
		}));
		test_nbody_engine	tc1(nbody_create_engine(param));
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap			param(std::map<QString, QVariant>({{"engine", "opencl_bh"},
			{"verbose", "1"},
			{"distance_to_node_radius_ratio", 1e8},
			{"traverse_type", "nested_tree"}
		}));
		test_nbody_engine	tc1(nbody_create_engine(param));
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param1(std::map<QString, QVariant>({{"engine", "opencl_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "cycle"},
			{"tree_layout", "heap"}
		}));
		QVariantMap param2(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "cycle"},
			{"tree_layout", "heap"}
		}));
		test_nbody_engine_compare tc1(nbody_create_engine(param1),
									  nbody_create_engine(param2),
									  256, 1e-12);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param1(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "nested_tree"},
			{"tree_layout", "heap"}
		}));
		QVariantMap param2(std::map<QString, QVariant>({{"engine", "opencl_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "nested_tree"},
			{"tree_layout", "heap"}
		}));
		test_nbody_engine_compare tc1(nbody_create_engine(param1),
									  nbody_create_engine(param2),
									  128, 1e-13);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param1(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "cycle"},
			{"tree_layout", "heap"}
		}));
		QVariantMap param2(std::map<QString, QVariant>({{"engine", "opencl_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "cycle"},
			{"tree_layout", "heap"}
		}));
		test_nbody_engine_compare tc1(nbody_create_engine(param1),
									  nbody_create_engine(param2),
									  128, 1e-13);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param1(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "nested_tree"},
			{"tree_layout", "heap"},
			{"tree_build_rate", 2}
		}));
		QVariantMap param2(std::map<QString, QVariant>({{"engine", "opencl_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "nested_tree"},
			{"tree_layout", "heap"},
			{"tree_build_rate", 2}
		}));
		test_nbody_engine_compare tc1(nbody_create_engine(param1),
									  nbody_create_engine(param2),
									  128, 1.6874e-13, 2);
		res += QTest::qExec(&tc1, argc, argv);
	}
	{
		QVariantMap param1(std::map<QString, QVariant>({{"engine", "simple_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "cycle"},
			{"tree_layout", "heap"},
			{"tree_build_rate", 2}
		}));
		QVariantMap param2(std::map<QString, QVariant>({{"engine", "opencl_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", "cycle"},
			{"tree_layout", "heap"},
			{"tree_build_rate", 2}
		}));
		test_nbody_engine_compare tc1(nbody_create_engine(param1),
									  nbody_create_engine(param2),
									  128, 1.6874e-13, 2);
		res += QTest::qExec(&tc1, argc, argv);
	}
	for(const char* traverse_type : {"cycle", "nested_tree"})
	{
		QVariantMap param1(std::map<QString, QVariant>({{"engine", "opencl_bh"},
			{"distance_to_node_radius_ratio", 3.1623},
			{"traverse_type", traverse_type},
			{"tree_layout", "heap"},
			{"tree_build_rate", 2}
		}));
		QVariantMap param2(param1); param2["tree_layout"] = "heap_stackless";
		test_nbody_engine_compare tc1(nbody_create_engine(param1),
									  nbody_create_engine(param2),
									  128, 2.84e-13, 2);
		res += QTest::qExec(&tc1, argc, argv);
	}
	for(const char* device : {"0:0;0:0", "0:0,0", "0:0,0,0,0", "0:0,0,0,0,0,0,0,0"})
	{
		for(const char* traverse_type : {"cycle", "nested_tree"})
		{
			for(const char* tree_layout : {"heap", "heap_stackless"})
			{
				for(int tree_build_rate : {0, 2})
				{
					QVariantMap param1(std::map<QString, QVariant>({{"engine", "opencl_bh"},
						{"device", "0:0"},
						{"block_size", 16},
						{"distance_to_node_radius_ratio", 3.1623},
						{"traverse_type", traverse_type},
						{"tree_layout", tree_layout},
						{"tree_build_rate", tree_build_rate}
					}));
					QVariantMap param2(param1); param2["device"] = device;
					test_nbody_engine_compare tc1(nbody_create_engine(param1),
												  nbody_create_engine(param2),
												  64, 1e-16, 2);
					res += QTest::qExec(&tc1, argc, argv);
					if(res > 0)
					{
						return res;
					}
				}
			}
		}
	}
	return res;
}
#endif // HAVE_OPENCL
int main(int argc, char* argv[])
{
	int res = 0;

	res += common_test(argc, argv);
	res += ah_test(argc, argv);
	res += block_test(argc, argv);
#ifdef HAVE_CUDA
	res += cuda_test(argc, argv);
#endif //HAVE_CUDA
	res += openmp_test(argc, argv);
	res += simple_test(argc, argv);
	res += simple_bh_test(argc, argv);
#ifdef HAVE_OPENCL
	res += opencl_test(argc, argv);
#endif // HAVE_OPENCL

	return res;
}

#include "test_nbody_engine.moc"
