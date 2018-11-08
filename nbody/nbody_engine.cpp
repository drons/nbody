#include "nbody_engine.h"

nbody_engine::memory::memory()
{

}

nbody_engine::memory::~memory()
{

}

nbody_engine::nbody_engine()
{
	m_compute_count = 0;
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
