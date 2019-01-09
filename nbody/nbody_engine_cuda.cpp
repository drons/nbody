#include "nbody_engine_cuda.h"

#include <QDebug>

#include "nbody_engine_cuda_memory.h"

nbody_engine_cuda::nbody_engine_cuda() :
	m_mass(NULL),
	m_y(NULL),
	m_data(NULL),
	m_block_size(NBODY_DATA_BLOCK_SIZE)
{
}

nbody_engine_cuda::~nbody_engine_cuda()
{
	delete m_mass;
	delete m_y;
}

const char* nbody_engine_cuda::type_name() const
{
	return "nbody_engine_cuda";
}

void nbody_engine_cuda::init(nbody_data* body_data)
{
	m_data = body_data;
	m_mass = dynamic_cast<smemory*>(create_buffer(sizeof(nbcoord_t) * m_data->get_count()));
	m_y = dynamic_cast<smemory*>(create_buffer(sizeof(nbcoord_t) * problem_size()));

	std::vector<nbcoord_t>	ytmp(problem_size());
	size_t					count = m_data->get_count();
	nbcoord_t*				rx = ytmp.data();
	nbcoord_t*				ry = rx + count;
	nbcoord_t*				rz = rx + 2 * count;
	nbcoord_t*				vx = rx + 3 * count;
	nbcoord_t*				vy = rx + 4 * count;
	nbcoord_t*				vz = rx + 5 * count;
	const nbvertex_t*		vrt = m_data->get_vertites();
	const nbvertex_t*		vel = m_data->get_velosites();

	for(size_t i = 0; i != count; ++i)
	{
		rx[i] = vrt[i].x;
		ry[i] = vrt[i].y;
		rz[i] = vrt[i].z;
		vx[i] = vel[i].x;
		vy[i] = vel[i].y;
		vz[i] = vel[i].z;
	}

	write_buffer(m_mass, const_cast<nbcoord_t*>(m_data->get_mass()));
	write_buffer(m_y, ytmp.data());
}

void nbody_engine_cuda::get_data(nbody_data* body_data)
{
	std::vector<nbcoord_t>	ytmp(problem_size());
	size_t					count = body_data->get_count();
	const nbcoord_t*		rx = ytmp.data();
	const nbcoord_t*		ry = rx + count;
	const nbcoord_t*		rz = rx + 2 * count;
	const nbcoord_t*		vx = rx + 3 * count;
	const nbcoord_t*		vy = rx + 4 * count;
	const nbcoord_t*		vz = rx + 5 * count;
	nbvertex_t*				vrt = body_data->get_vertites();
	nbvertex_t*				vel = body_data->get_velosites();

	read_buffer(ytmp.data(), m_y);

	for(size_t i = 0; i != count; ++i)
	{
		vrt[i].x = rx[i];
		vrt[i].y = ry[i];
		vrt[i].z = rz[i];
		vel[i].x = vx[i];
		vel[i].y = vy[i];
		vel[i].z = vz[i];
	}
}

size_t nbody_engine_cuda::problem_size() const
{
	return m_data->get_count() * 6;
}

nbody_engine::memory* nbody_engine_cuda::get_y()
{
	return m_y;
}

void nbody_engine_cuda::advise_time(const nbcoord_t& dt)
{
	m_data->advise_time(dt);
}

nbcoord_t nbody_engine_cuda::get_time() const
{
	return m_data->get_time();
}

void nbody_engine_cuda::set_time(nbcoord_t t)
{
	m_data->set_time(t);
}

size_t nbody_engine_cuda::get_step() const
{
	return m_data->get_step();
}

void nbody_engine_cuda::set_step(size_t s)
{
	m_data->set_step(s);
}

void nbody_engine_cuda::fcompute(const nbcoord_t& t, const memory* _y, memory* _f)
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

	size_t			count = m_data->get_count();

	fcompute_block(static_cast<const nbcoord_t*>(y->data()), static_cast<nbcoord_t*>(f->data()),
				   static_cast<const nbcoord_t*>(m_mass->data()),
				   static_cast<int>(count), get_block_size());
	fcompute_xyz(static_cast<const nbcoord_t*>(y->data()), static_cast<nbcoord_t*>(f->data()),
				 static_cast<int>(count), get_block_size());
}

nbody_engine_cuda::memory* nbody_engine_cuda::create_buffer(size_t s)
{
	smemory*	mem = new smemory(s);
	if(mem->size() == 0 && s > 0)
	{
		delete mem;
		return NULL;
	}
	return mem;
}

void nbody_engine_cuda::free_buffer(memory* mem)
{
	delete mem;
}

void nbody_engine_cuda::read_buffer(void* dst, const memory* _src)
{
	const smemory*	src = dynamic_cast<const smemory*>(_src);

	if(src == NULL)
	{
		qDebug() << "src is not smemory";
		return;
	}

	cudaMemcpy(dst, src->data(), src->size(), cudaMemcpyDeviceToHost);
}

void nbody_engine_cuda::write_buffer(memory* _dst, const void* src)
{
	smemory*	dst = dynamic_cast<smemory*>(_dst);

	if(dst == NULL)
	{
		qDebug() << "dst is not smemory";
		return;
	}

	cudaMemcpy(dst->data(), src, dst->size(), cudaMemcpyHostToDevice);
}

void nbody_engine_cuda::copy_buffer(memory* _a, const memory* _b)
{
	smemory*		a = dynamic_cast<smemory*>(_a);
	const smemory*	b = dynamic_cast<const smemory*>(_b);

	if(a == NULL)
	{
		qDebug() << "a is not smemory";
		return;
	}
	if(b == NULL)
	{
		qDebug() << "b is not smemory";
		return;
	}
	if(a->size() != b->size())
	{
		qDebug() << "Size does not match";
		return;
	}

	cudaMemcpy(a->data(), b->data(), a->size(), cudaMemcpyHostToHost);
}

void nbody_engine_cuda::fill_buffer(memory* _a, const nbcoord_t& value)
{
	smemory*		a = dynamic_cast<smemory*>(_a);

	if(a == NULL)
	{
		qDebug() << "a is not smemory";
		return;
	}

	::fill_buffer(static_cast<nbcoord_t*>(a->data()), value,
				  static_cast<int>(a->size() / sizeof(nbcoord_t)));
}

void nbody_engine_cuda::fmadd_inplace(memory* _a, const memory* _b, const nbcoord_t& c)
{
	smemory*		a = dynamic_cast<smemory*>(_a);
	const smemory*	b = dynamic_cast<const smemory*>(_b);

	if(a == NULL)
	{
		qDebug() << "a is not smemory";
		return;
	}
	if(b == NULL)
	{
		qDebug() << "b is not smemory";
		return;
	}
	::fmadd_inplace(0, static_cast<nbcoord_t*>(a->data()),
					static_cast<const nbcoord_t*>(b->data()), c,
					static_cast<int>(a->size() / sizeof(nbcoord_t)));
}

void nbody_engine_cuda::fmadd(memory* _a, const memory* _b, const memory* _c, const nbcoord_t& d)
{
	smemory*		a = dynamic_cast<smemory*>(_a);
	const smemory*	b = dynamic_cast<const smemory*>(_b);
	const smemory*	c = dynamic_cast<const smemory*>(_c);

	if(a == NULL)
	{
		qDebug() << "a is not smemory";
		return;
	}
	if(b == NULL)
	{
		qDebug() << "b is not smemory";
		return;
	}
	if(c == NULL)
	{
		qDebug() << "c is not smemory";
		return;
	}

	::fmadd(static_cast<nbcoord_t*>(a->data()), static_cast<const nbcoord_t*>(b->data()),
			static_cast<const nbcoord_t*>(c->data()), d, 0, 0, 0,
			static_cast<int>(a->size() / sizeof(nbcoord_t)));
}

void nbody_engine_cuda::fmaxabs(const nbody_engine::memory* _a, nbcoord_t& result)
{
	const smemory*	a = dynamic_cast<const smemory*>(_a);

	if(a == NULL)
	{
		qDebug() << "a is not smemory";
		return;
	}

	int		size(static_cast<int>(a->size() / sizeof(nbcoord_t)));

	if(size == 0)
	{
		result = 0;
		return;
	}

	::fmaxabs(static_cast<const nbcoord_t*>(a->data()), size, result);
}

void nbody_engine_cuda::print_info() const
{
	cudaDeviceProp	prop;
	memset(&prop, 0, sizeof(prop));
	cudaGetDeviceProperties(&prop, 0);
	qDebug() << "\t" << prop.name;
	qDebug() << "\t" << "Global memory" << prop.totalGlobalMem;
	qDebug() << "\t" << "Const memory " << prop.totalConstMem;
	qDebug() << "\t" << "Shared memory" << prop.sharedMemPerBlock << "/" << prop.sharedMemPerMultiprocessor;
	qDebug() << "\t" << "Compute Cap  " << prop.major << "." << prop.minor;
	qDebug() << "\t" << "Block size   " << m_block_size;
}

int nbody_engine_cuda::get_block_size() const
{
	return m_block_size;
}

void nbody_engine_cuda::set_block_size(int block_size)
{
	m_block_size = block_size;
}
