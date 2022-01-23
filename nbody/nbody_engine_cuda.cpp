#include "nbody_engine_cuda.h"

#include <QDebug>
#include <omp.h>

#include "nbody_engine_cuda_memory.h"

struct nbody_engine_cuda::data
{
	smemory*			m_mass;
	smemory*			m_y;
	nbody_data*			m_data;
	int					m_block_size;
	std::vector<int>	m_device_ids;
	data():
		m_mass(nullptr),
		m_y(nullptr),
		m_data(nullptr),
		m_block_size(NBODY_DATA_BLOCK_SIZE),
		m_device_ids(1, 0)
	{}
};

nbody_engine_cuda::nbody_engine_cuda() :
	d(new data())
{
}

nbody_engine_cuda::~nbody_engine_cuda()
{
	delete d->m_mass;
	delete d->m_y;
	delete d;
}

const char* nbody_engine_cuda::type_name() const
{
	return "nbody_engine_cuda";
}

bool nbody_engine_cuda::init(nbody_data* body_data)
{
	d->m_data = body_data;
	d->m_mass = dynamic_cast<smemory*>(create_buffer(sizeof(nbcoord_t) * d->m_data->get_count()));
	d->m_y = dynamic_cast<smemory*>(create_buffer(sizeof(nbcoord_t) * problem_size()));
	if(d->m_mass == nullptr || d->m_y == nullptr)
	{
		return false;
	}
	std::vector<nbcoord_t>	ytmp(problem_size());
	size_t					count = d->m_data->get_count();
	nbcoord_t*				rx = ytmp.data();
	nbcoord_t*				ry = rx + count;
	nbcoord_t*				rz = rx + 2 * count;
	nbcoord_t*				vx = rx + 3 * count;
	nbcoord_t*				vy = rx + 4 * count;
	nbcoord_t*				vz = rx + 5 * count;
	const nbvertex_t*		vrt = d->m_data->get_vertites();
	const nbvertex_t*		vel = d->m_data->get_velosites();

	for(size_t i = 0; i != count; ++i)
	{
		rx[i] = vrt[i].x;
		ry[i] = vrt[i].y;
		rz[i] = vrt[i].z;
		vx[i] = vel[i].x;
		vy[i] = vel[i].y;
		vz[i] = vel[i].z;
	}

	write_buffer(d->m_mass, const_cast<nbcoord_t*>(d->m_data->get_mass()));
	write_buffer(d->m_y, ytmp.data());

	return true;
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

	read_buffer(ytmp.data(), d->m_y);

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
	return d->m_data->get_count() * 6;
}

nbody_engine::memory* nbody_engine_cuda::get_y()
{
	return d->m_y;
}

void nbody_engine_cuda::advise_time(const nbcoord_t& dt)
{
	d->m_data->advise_time(dt);
}

nbcoord_t nbody_engine_cuda::get_time() const
{
	return d->m_data->get_time();
}

void nbody_engine_cuda::set_time(nbcoord_t t)
{
	d->m_data->set_time(t);
}

size_t nbody_engine_cuda::get_step() const
{
	return d->m_data->get_step();
}

void nbody_engine_cuda::set_step(size_t s)
{
	d->m_data->set_step(s);
}

void nbody_engine_cuda::fcompute(const nbcoord_t& t, const memory* _y, memory* _f)
{
	Q_UNUSED(t);
	const smemory*	y = dynamic_cast<const  smemory*>(_y);
	smemory*		f = dynamic_cast<smemory*>(_f);

	if(y == nullptr)
	{
		qDebug() << "y is not smemory";
		return;
	}
	if(f == nullptr)
	{
		qDebug() << "f is not smemory";
		return;
	}

	advise_compute_count();
	if(d->m_device_ids.size() > 1)
	{
		synchronize_y(y);
	}

	size_t	count = d->m_data->get_count();
	#pragma omp parallel num_threads(d->m_device_ids.size())
	{
		size_t	dev_n = static_cast<size_t>(omp_get_thread_num());
		size_t	dev_count(count / d->m_device_ids.size());
		size_t	dev_off(dev_count * dev_n);
		cudaSetDevice(d->m_device_ids[dev_n]);

		fcompute_block(dev_off, static_cast<const nbcoord_t*>(y->data(dev_n)),
					   static_cast<nbcoord_t*>(f->data(dev_n)),
					   static_cast<const nbcoord_t*>(d->m_mass->data(dev_n)),
					   dev_count, count, get_block_size());
		fcompute_xyz(static_cast<const nbcoord_t*>(y->data(dev_n)) + dev_off,
					 static_cast<nbcoord_t*>(f->data(dev_n)) + dev_off,
					 dev_count, count, get_block_size());
		cudaDeviceSynchronize();
	}

	if(d->m_device_ids.size() > 1)
	{
		synchronize_f(f);
	}
}

nbody_engine_cuda::memory* nbody_engine_cuda::create_buffer(size_t s)
{
	smemory*	mem = new smemory(s, d->m_device_ids);
	if(mem->size() == 0 && s > 0)
	{
		delete mem;
		return nullptr;
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

	if(src == nullptr)
	{
		qDebug() << "src is not smemory";
		return;
	}

	size_t		size = src->size();
	size_t		dev_size = size / d->m_device_ids.size();
	#pragma omp parallel num_threads(d->m_device_ids.size())
	{
		size_t	dev_n = static_cast<size_t>(omp_get_thread_num());
		size_t	dev_off(dev_size * dev_n);
		cudaSetDevice(d->m_device_ids[dev_n]);
		cudaMemcpy(static_cast<char*>(dst) + dev_off,
				   static_cast<const char*>(src->data(dev_n)) + dev_off,
				   dev_size, cudaMemcpyDeviceToHost);
	}
}

void nbody_engine_cuda::write_buffer(memory* _dst, const void* src)
{
	smemory*	dst = dynamic_cast<smemory*>(_dst);

	if(dst == nullptr)
	{
		qDebug() << "dst is not smemory";
		return;
	}

	#pragma omp parallel num_threads(d->m_device_ids.size())
	{
		size_t	dev_n = static_cast<size_t>(omp_get_thread_num());
		cudaSetDevice(d->m_device_ids[dev_n]);
		cudaMemcpy(dst->data(dev_n), src, dst->size(), cudaMemcpyHostToDevice);
	}
}

void nbody_engine_cuda::copy_buffer(memory* _a, const memory* _b)
{
	smemory*		a = dynamic_cast<smemory*>(_a);
	const smemory*	b = dynamic_cast<const smemory*>(_b);

	if(a == nullptr)
	{
		qDebug() << "a is not smemory";
		return;
	}
	if(b == nullptr)
	{
		qDebug() << "b is not smemory";
		return;
	}
	if(a->size() != b->size())
	{
		qDebug() << "Size does not match";
		return;
	}

	#pragma omp parallel num_threads(d->m_device_ids.size())
	{
		size_t	dev_n = static_cast<size_t>(omp_get_thread_num());
		cudaSetDevice(d->m_device_ids[dev_n]);
		cudaMemcpy(a->data(dev_n), b->data(dev_n), a->size(), cudaMemcpyHostToHost);
	}
}

void nbody_engine_cuda::fill_buffer(memory* _a, const nbcoord_t& value)
{
	smemory*		a = dynamic_cast<smemory*>(_a);

	if(a == nullptr)
	{
		qDebug() << "a is not smemory";
		return;
	}

	#pragma omp parallel num_threads(d->m_device_ids.size())
	{
		size_t	dev_n = static_cast<size_t>(omp_get_thread_num());
		cudaSetDevice(d->m_device_ids[dev_n]);
		::fill_buffer(static_cast<nbcoord_t*>(a->data(dev_n)), value,
					  static_cast<int>(a->size() / sizeof(nbcoord_t)));
	}
}

void nbody_engine_cuda::fmadd_inplace(memory* _a, const memory* _b, const nbcoord_t& c)
{
	smemory*		a = dynamic_cast<smemory*>(_a);
	const smemory*	b = dynamic_cast<const smemory*>(_b);

	if(a == nullptr)
	{
		qDebug() << "a is not smemory";
		return;
	}
	if(b == nullptr)
	{
		qDebug() << "b is not smemory";
		return;
	}
	size_t		count = a->size() / sizeof(nbcoord_t);
	size_t		dev_count = count / d->m_device_ids.size();
	#pragma omp parallel num_threads(d->m_device_ids.size())
	{
		size_t	dev_n = static_cast<size_t>(omp_get_thread_num());
		size_t	dev_off(dev_count * dev_n);
		cudaSetDevice(d->m_device_ids[dev_n]);
		::fmadd_inplace(static_cast<nbcoord_t*>(a->data(dev_n)) + dev_off,
						static_cast<const nbcoord_t*>(b->data(dev_n)) + dev_off, c,
						dev_count);
	}
}

void nbody_engine_cuda::fmaddn_corr(nbody_engine::memory* _a, nbody_engine::memory* _corr,
									const nbody_engine::memory_array& _b, const nbcoord_t* c, size_t csize)
{
	smemory*		a = dynamic_cast<smemory*>(_a);
	smemory*		corr = dynamic_cast<smemory*>(_corr);

	if(a == nullptr)
	{
		qDebug() << "a is not smemory";
		return;
	}
	if(corr == nullptr)
	{
		qDebug() << "corr is not smemory";
		return;
	}
	if(c == nullptr)
	{
		qDebug() << "c must not be nullptr";
		return;
	}
	if(csize > _b.size())
	{
		qDebug() << "csize > b.size()";
		return;
	}
	for(size_t k = 0; k < csize; ++k)
	{
		const smemory* b = dynamic_cast<const smemory*>(_b[k]);
		if(b == nullptr)
		{
			qDebug() << "b is not smemory";
			return;
		}
	}
	size_t		count = a->size() / sizeof(nbcoord_t);
	size_t		dev_count = count / d->m_device_ids.size();
	#pragma omp parallel num_threads(d->m_device_ids.size())
	{
		size_t	dev_n = static_cast<size_t>(omp_get_thread_num());
		size_t	dev_off(dev_count * dev_n);
		cudaSetDevice(d->m_device_ids[dev_n]);
		for(size_t k = 0; k < csize; ++k)
		{
			if(c[k] == 0_f)
			{
				continue;
			}
			const smemory* b = dynamic_cast<const smemory*>(_b[k]);
			if(b == nullptr) {continue;} //Never hit here
			::fmadd_inplace_corr(static_cast<nbcoord_t*>(a->data(dev_n)) + dev_off,
								 static_cast<nbcoord_t*>(corr->data(dev_n)) + dev_off,
								 static_cast<const nbcoord_t*>(b->data(dev_n)) + dev_off, c[k],
								 dev_count);
		}
	}
}

void nbody_engine_cuda::fmadd(memory* _a, const memory* _b,
							  const memory* _c, const nbcoord_t& _d)
{
	smemory*		a = dynamic_cast<smemory*>(_a);
	const smemory*	b = dynamic_cast<const smemory*>(_b);
	const smemory*	c = dynamic_cast<const smemory*>(_c);

	if(a == nullptr)
	{
		qDebug() << "a is not smemory";
		return;
	}
	if(b == nullptr)
	{
		qDebug() << "b is not smemory";
		return;
	}
	if(c == nullptr)
	{
		qDebug() << "c is not smemory";
		return;
	}

	size_t		count = a->size() / sizeof(nbcoord_t);
	size_t		dev_count = count / d->m_device_ids.size();
	#pragma omp parallel num_threads(d->m_device_ids.size())
	{
		size_t	dev_n = static_cast<size_t>(omp_get_thread_num());
		size_t	dev_off(dev_count * dev_n);
		cudaSetDevice(d->m_device_ids[dev_n]);
		::fmadd(static_cast<nbcoord_t*>(a->data(dev_n)) + dev_off,
				static_cast<const nbcoord_t*>(b->data(dev_n)) + dev_off,
				static_cast<const nbcoord_t*>(c->data(dev_n)) + dev_off,
				_d, dev_count);
	}
}

void nbody_engine_cuda::fmaxabs(const nbody_engine::memory* _a, nbcoord_t& result)
{
	const smemory*	a = dynamic_cast<const smemory*>(_a);

	if(a == nullptr)
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

	std::vector<nbcoord_t>	devmax(d->m_device_ids.size(), 0);
	size_t		count = a->size() / sizeof(nbcoord_t);
	size_t		dev_count = count / d->m_device_ids.size();
	#pragma omp parallel num_threads(d->m_device_ids.size())
	{
		size_t	dev_n = static_cast<size_t>(omp_get_thread_num());
		size_t	dev_off(dev_count * dev_n);
		cudaSetDevice(d->m_device_ids[dev_n]);
		::fmaxabs(static_cast<const nbcoord_t*>(a->data(dev_n)) + dev_off,
				  dev_count, devmax[dev_n]);
	}

	result = *std::max_element(devmax.begin(), devmax.end());
}

void nbody_engine_cuda::print_info() const
{
	qDebug() << "\tSelected CUDA devices:";
	for(size_t n = 0; n != d->m_device_ids.size(); ++n)
	{
		cudaDeviceProp	prop;
		memset(&prop, 0, sizeof(prop));
		cudaGetDeviceProperties(&prop, d->m_device_ids[n]);
		qDebug() << "\t #" << n << "ID" << d->m_device_ids[n] << prop.name << "PCI Bus" << prop.pciBusID
				 << "Id" << prop.pciDeviceID << "Domain" << prop.pciDomainID
#if CUDART_VERSION >= 10000
				 << "UUID" << QByteArray(prop.uuid.bytes, sizeof(prop.uuid.bytes)).toHex().data()
#endif//CUDART_VERSION >= 10000
				 << "";
		qDebug() << "\t\t" << "Memory (Global/Const/L2)"
				 << prop.totalGlobalMem / (1024 * 1024) << "MB /"
				 << prop.totalConstMem / 1024 << "KB /"
				 << prop.l2CacheSize / 1024 << "KB";
		qDebug() << "\t\t" << "Shared memory (block/multiprocessor)"
				 << prop.sharedMemPerBlock << "/" << prop.sharedMemPerMultiprocessor;
		qDebug() << "\t\t" << "Registers (block/multiprocessor)"
				 << prop.regsPerBlock << "/" << prop.regsPerMultiprocessor;
		qDebug() << "\t\t" << "Compute Cap  " << prop.major << "." << prop.minor;
		qDebug() << "\t\t" << "Max mem pitch" << prop.memPitch;
	}
	qDebug() << "\t" << "Block size   " << d->m_block_size;
}

int nbody_engine_cuda::get_block_size() const
{
	return d->m_block_size;
}

void nbody_engine_cuda::set_block_size(int block_size)
{
	d->m_block_size = block_size;
}

int nbody_engine_cuda::select_devices(const QString& devices_str)
{
	QStringList	dev_list(devices_str.split(",", QString::SkipEmptyParts));
	if(dev_list.isEmpty())
	{
		qDebug() << "CUDA device list is empty";
		return -1;
	}

	int			device_count = -1;
	cudaError_t	res = cudaGetDeviceCount(&device_count);
	if(res != cudaSuccess)
	{
		qDebug() << "No CUDA devices found";
		return -1;
	}

	std::vector<int>	device_ids;
	for(int i = 0; i != dev_list.size(); ++i)
	{
		bool	ok = false;
		int		dev_id = dev_list[i].toInt(&ok);

		if(!ok)
		{
			qDebug() << "Can't parse device ID" <<  dev_list[i];
			return -1;
		}

		if(dev_id < 0 || dev_id >= device_count)
		{
			qDebug() << "Invalid device ID" << dev_id << "must be in range [0 ..." << device_count << ")";
			return -1;
		}

		device_ids.push_back(dev_id);
	}

	d->m_device_ids = device_ids;
	return 0;
}

void nbody_engine_cuda::synchronize_y(const smemory* y)
{
	QByteArray	host_buffer(static_cast<int>(y->size()), Qt::Uninitialized);
	read_buffer(host_buffer.data(), y);
	write_buffer(const_cast<smemory*>(y), host_buffer.data());
}

void nbody_engine_cuda::synchronize_f(smemory* f)
{
	QByteArray	host_buffer(static_cast<int>(f->size()), Qt::Uninitialized);
	#pragma omp parallel num_threads(d->m_device_ids.size())
	{
		size_t	dev_n = static_cast<size_t>(omp_get_thread_num());
		size_t	device_data_size(f->size() / d->m_device_ids.size());
		size_t	row_count = 6;
		size_t	pitch = f->size() / row_count;
		size_t	width = device_data_size / row_count;
		size_t	dev_off(width * dev_n);
		cudaSetDevice(d->m_device_ids[dev_n]);
		cudaMemcpy2D(host_buffer.data() + dev_off, pitch,
					 static_cast<const char*>(f->data(dev_n)) + dev_off, pitch,
					 width, row_count, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}
	write_buffer(f, host_buffer.data());
}

void nbody_engine_cuda::synchronize_sum(nbody_engine_cuda::smemory* f)
{
	size_t	size = f->size();
	size_t	device_count = d->m_device_ids.size();
	std::vector<std::vector<nbcoord_t>>	host_buffers(device_count);

	#pragma omp parallel num_threads(device_count)
	{
		size_t	dev_n = static_cast<size_t>(omp_get_thread_num());
		host_buffers[dev_n].resize(size);
		cudaSetDevice(d->m_device_ids[dev_n]);
		cudaMemcpy(host_buffers[dev_n].data(), f->data(dev_n), size, cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}

	#pragma omp parallel for
	for(size_t i = 0; i < size; ++i)
	{
		for(size_t dev_n = 1; dev_n < device_count; ++dev_n)
		{
			host_buffers[0][i] += host_buffers[dev_n][i];
		}
	}
	write_buffer(f, host_buffers[0].data());
}

const std::vector<int>& nbody_engine_cuda::get_device_ids() const
{
	return d->m_device_ids;
}

const nbody_data* nbody_engine_cuda::get_data() const
{
	return d->m_data;
}

const nbody_engine_cuda::smemory* nbody_engine_cuda::get_mass() const
{
	return d->m_mass;
}

