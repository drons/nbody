#include "nbody_engine.h"

nbody_engine::memory::memory()
{
}

nbody_engine::memory::~memory()
{
}

nbody_engine::nbody_engine():
	m_compute_count(0),
	m_ode_order(eode_first_order)
{
}

nbody_engine::~nbody_engine()
{
}

nbody_engine::memory_array nbody_engine::create_buffers(size_t size, size_t count)
{
	memory_array	mema;
	mema.reserve(count);
	for(size_t n = 0; n != count; ++n)
	{
		memory*	m = create_buffer(size);
		if(m == NULL)
		{
			free_buffers(mema);
			return memory_array();
		}
		mema.push_back(m);
	}
	return mema;
}

void nbody_engine::free_buffers(memory_array& mema)
{
	for(memory* m : mema)
	{
		free_buffer(m);
	}
	mema.clear();
}

void nbody_engine::fmaddn_inplace(memory* a, const memory_array& b, const nbcoord_t* c)
{
	if(c == NULL)
	{
		return;
	}
	for(size_t i = 0; i != b.size(); ++i)
	{
		fmadd(a, a, b[i], c[i]);
	}
}

//! a[i] = b[i] + sum( c[k][i]*d[k], k=[0...c.size()) )
void nbody_engine::fmaddn(memory* a, const memory* b, const memory_array& c,
						  const nbcoord_t* d, size_t dsize)
{
	if(d == NULL)
	{
		qDebug() << "dsize > c.size()";
		return;
	}
	if(dsize > c.size())
	{
		qDebug() << "dsize > c.size()";
		return;
	}

	if(b == NULL)
	{
		fill_buffer(a, 0);
	}

	for(size_t i = 0; i != dsize; ++i)
	{
		if(i == 0 && b != NULL)
		{
			fmadd(a, b, c[i], d[i]);
		}
		else
		{
			fmadd(a, a, c[i], d[i]);
		}
	}
}

void nbody_engine::print_info() const
{
}

void nbody_engine::advise_compute_count()
{
	m_compute_count++;
}

size_t nbody_engine::get_compute_count() const
{
	return m_compute_count;
}

void nbody_engine::set_ode_order(e_ode_order order)
{
	m_ode_order = order;
}

e_ode_order nbody_engine::get_ode_order() const
{
	return m_ode_order;
}
