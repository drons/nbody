#include "nbody_engine_openmp.h"
#include <QDebug>
#include <omp.h>
#include "summation.h"

nbody_engine_openmp::nbody_engine_openmp()
{
}

nbody_engine_openmp::~nbody_engine_openmp()
{
}

const char* nbody_engine_openmp::type_name() const
{
	return "nbody_engine_openmp";
}

void nbody_engine_openmp::fcompute(const nbcoord_t& t, const memory* _y, memory* _f)
{
	Q_UNUSED(t);
	const smemory*	y = dynamic_cast<const  smemory*>(_y);
	smemory*		f = dynamic_cast<smemory*>(_f);

	if(y == NULL)
	{
		qDebug() << "y is not smemory";
		return;
	}
	if(f == NULL)
	{
		qDebug() << "f is not smemory";
		return;
	}

	advise_compute_count();

	size_t				count = m_data->get_count();
	const nbcoord_t*	rx = reinterpret_cast<const nbcoord_t*>(y->data());
	const nbcoord_t*	ry = rx + count;
	const nbcoord_t*	rz = rx + 2 * count;
	const nbcoord_t*	vx = rx + 3 * count;
	const nbcoord_t*	vy = rx + 4 * count;
	const nbcoord_t*	vz = rx + 5 * count;

	nbcoord_t*			frx = reinterpret_cast<nbcoord_t*>(f->data());
	nbcoord_t*			fry = frx + count;
	nbcoord_t*			frz = frx + 2 * count;
	nbcoord_t*			fvx = frx + 3 * count;
	nbcoord_t*			fvy = frx + 4 * count;
	nbcoord_t*			fvz = frx + 5 * count;

	const nbcoord_t*	mass = reinterpret_cast<const nbcoord_t*>(m_mass->data());

	#pragma omp parallel for
	for(size_t body1 = 0; body1 < count; ++body1)
	{
		const nbvertex_t	v1(rx[ body1 ], ry[ body1 ], rz[ body1 ]);
		nbvertex_t			total_force;
		for(size_t body2 = 0; body2 != count; ++body2)
		{
			if(body1 == body2)
			{
				continue;
			}
			const nbvertex_t	v2(rx[ body2 ], ry[ body2 ], rz[ body2 ]);
			const nbvertex_t	force(m_data->force(v1, v2, mass[body1], mass[body2]));
			total_force += force;
		}
		frx[body1] = vx[body1];
		fry[body1] = vy[body1];
		frz[body1] = vz[body1];
		fvx[body1] = total_force.x / mass[body1];
		fvy[body1] = total_force.y / mass[body1];
		fvz[body1] = total_force.z / mass[body1];
	}
}

void nbody_engine_openmp::copy_buffer(nbody_engine::memory* __a, const nbody_engine::memory* __b)
{
	smemory*			_a = dynamic_cast<smemory*>(__a);
	const smemory*		_b = dynamic_cast<const smemory*>(__b);

	if(_a == NULL)
	{
		qDebug() << "a is not smemory";
		return;
	}

	if(_b == NULL)
	{
		qDebug() << "b is not smemory";
		return;
	}

	nbcoord_t*			a = reinterpret_cast<nbcoord_t*>(_a->data());
	const nbcoord_t*	b = reinterpret_cast<const nbcoord_t*>(_b->data());
	size_t				count = problem_size();

	#pragma omp parallel for
	for(size_t i = 0; i < count; ++i)
	{
		a[i] = b[i];
	}
}

void nbody_engine_openmp::fill_buffer(nbody_engine::memory* __a, const nbcoord_t& value)
{
	smemory*			_a = dynamic_cast<smemory*>(__a);

	if(_a == NULL)
	{
		qDebug() << "a is not smemory";
		return;
	}

	nbcoord_t*			a = reinterpret_cast<nbcoord_t*>(_a->data());
	size_t				count = _a->size() / sizeof(nbcoord_t);

	#pragma omp parallel for
	for(size_t i = 0; i < count; ++i)
	{
		a[i] = value;
	}
}

void nbody_engine_openmp::fmadd_inplace(memory* __a, const memory* __b, const nbcoord_t& c)
{
	smemory*			_a = dynamic_cast<smemory*>(__a);
	const smemory*		_b = dynamic_cast<const smemory*>(__b);

	if(_a == NULL)
	{
		qDebug() << "a is not smemory";
		return;
	}
	if(_b == NULL)
	{
		qDebug() << "b is not smemory";
		return;
	}
	nbcoord_t*			a = reinterpret_cast<nbcoord_t*>(_a->data());
	const nbcoord_t*	b = reinterpret_cast<const nbcoord_t*>(_b->data());
	size_t				count = problem_size();

	#pragma omp parallel for
	for(size_t i = 0; i < count; ++i)
	{
		a[i] += b[i] * c;
	}
}

void nbody_engine_openmp::fmadd(memory* __a, const memory* __b, const memory* __c, const nbcoord_t& d)
{
	smemory*			_a = dynamic_cast<smemory*>(__a);
	const smemory*		_b = dynamic_cast<const smemory*>(__b);
	const smemory*		_c = dynamic_cast<const smemory*>(__c);

	if(_a == NULL)
	{
		qDebug() << "a is not smemory";
		return;
	}
	if(_b == NULL)
	{
		qDebug() << "b is not smemory";
		return;
	}
	if(_c == NULL)
	{
		qDebug() << "c is not smemory";
		return;
	}

	nbcoord_t*			a = reinterpret_cast<nbcoord_t*>(_a->data());
	const nbcoord_t*	b = reinterpret_cast<const nbcoord_t*>(_b->data());
	const nbcoord_t*	c = reinterpret_cast<const nbcoord_t*>(_c->data());
	size_t				count = problem_size();

	#pragma omp parallel for
	for(size_t i = 0; i < count; ++i)
	{
		a[i] = b[i] + c[i] * d;
	}
}

void nbody_engine_openmp::fmaddn_corr(memory* __a, memory* __corr, const memory_array& __b, const nbcoord_t* c)
{
	smemory*	_a = dynamic_cast<smemory*>(__a);
	smemory*	_corr = dynamic_cast<smemory*>(__corr);
	if(_a == nullptr)
	{
		qDebug() << "a is not smemory";
		return;
	}
	if(_corr == nullptr)
	{
		qDebug() << "corr is not smemory";
		return;
	}
	if(c == nullptr)
	{
		qDebug() << "c is not smemory";
		return;
	}
	std::vector<const nbcoord_t*>	b;
	for(auto i : __b)
	{
		const smemory* _b = dynamic_cast<const smemory*>(i);
		if(_b == nullptr)
		{
			qDebug() << "b is not smemory";
			return;
		}
		b.push_back(reinterpret_cast<const nbcoord_t*>(_b->data()));
	}

	//! Use volatile to prevent over-optimization at summation_k
	volatile nbcoord_t*	a = reinterpret_cast<nbcoord_t*>(_a->data());
	volatile nbcoord_t*	corr = reinterpret_cast<nbcoord_t*>(_corr->data());
	size_t	count = problem_size();
	size_t	count_b = b.size();

	#pragma omp parallel for
	for(size_t i = 0; i < count; ++i)
	{
		for(size_t k = 0; k < count_b; ++k)
		{
			volatile nbcoord_t	term(b[k][i] * c[k]);
			a[i] = summation_k(a[i], term, corr[i]);
		}
	}
}

void nbody_engine_openmp::fmaxabs(const nbody_engine::memory* __a, nbcoord_t& result)
{
	const smemory*		_a = dynamic_cast<const smemory*>(__a);

	if(_a == NULL)
	{
		qDebug() << "a is not smemory";
		return;
	}

	const nbcoord_t*	a = reinterpret_cast<const nbcoord_t*>(_a->data());
	size_t				count = problem_size();

	result = fabs(a[0]);

#if __GNUC__*100 + __GNUC_MINOR__ >= 409
	#pragma omp parallel for reduction( max : result )
#endif // since gcc-4.9
	for(size_t n = 0; n < count; ++n)
	{
		nbcoord_t	v(fabs(a[n]));
		if(v > result)
		{
			result = v;
		}
	}
}

void nbody_engine_openmp::print_info() const
{
	qDebug() << "\tOpenMP max threads:" << omp_get_max_threads();
}
