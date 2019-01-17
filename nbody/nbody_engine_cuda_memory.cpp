#include "nbody_engine_cuda_memory.h"
#include <omp.h>

nbody_engine_cuda::smemory::smemory(size_t size, const std::vector<int>& dev_ids) :
	m_data(dev_ids.size(), NULL),
	m_tex(dev_ids.size(), 0),
	m_size(size),
	m_device_ids(dev_ids)
{
	#pragma omp parallel num_threads(m_device_ids.size())
	{
		size_t	dev_n = static_cast<size_t>(omp_get_thread_num());
		cudaSetDevice(m_device_ids[dev_n]);
		cudaError_t res = cudaMalloc(&m_data[dev_n], size);
		if(res != cudaSuccess)
		{
			qDebug() << "Can't alloc " << size << "bytes for dev_id " << m_device_ids[dev_n];
			qDebug() << "Err code" << res << cudaGetErrorString(res);
		}
	}
}

nbody_engine_cuda::smemory::~smemory()
{
	#pragma omp parallel num_threads(m_device_ids.size())
	{
		size_t	dev_n = static_cast<size_t>(omp_get_thread_num());
		cudaSetDevice(m_device_ids[dev_n]);
		if(m_tex[dev_n] != 0)
		{
			cudaDestroyTextureObject(m_tex[dev_n]);
		}
		cudaFree(m_data[dev_n]);
	}
}

const void* nbody_engine_cuda::smemory::data(size_t dev_id) const
{
	return m_data[dev_id];
}

void* nbody_engine_cuda::smemory::data(size_t dev_n)
{
	return m_data[dev_n];
}

template<class T>
bool setup_texture_type(cudaResourceDesc&, nbody_engine_cuda::smemory::evecsize)
{
	return false;
}

template<>
bool setup_texture_type<float>(cudaResourceDesc& res_desc, nbody_engine_cuda::smemory::evecsize vec_size)
{
	res_desc.res.linear.desc.f = cudaChannelFormatKindFloat;
	if(vec_size == nbody_engine_cuda::smemory::evs1)
	{
		res_desc.res.linear.desc.x = 32; // bits per channel
	}
	else if(vec_size == nbody_engine_cuda::smemory::evs4)
	{
		res_desc.res.linear.desc.x = 32; // bits per channel
		res_desc.res.linear.desc.y = 32; // bits per channel
		res_desc.res.linear.desc.z = 32; // bits per channel
		res_desc.res.linear.desc.w = 32; // bits per channel
	}
	else
	{
		return false;
	}
	return true;
}

template<>
bool setup_texture_type<double>(cudaResourceDesc& res_desc, nbody_engine_cuda::smemory::evecsize vec_size)
{
	res_desc.res.linear.desc.f = cudaChannelFormatKindSigned;
	if(vec_size == nbody_engine_cuda::smemory::evs1)
	{
		res_desc.res.linear.desc.x = 32; // bits per channel
		res_desc.res.linear.desc.y = 32; // bits per channel
	}
	else if(vec_size == nbody_engine_cuda::smemory::evs4)
	{
		res_desc.res.linear.desc.x = 32; // bits per channel
		res_desc.res.linear.desc.y = 32; // bits per channel
		res_desc.res.linear.desc.z = 32; // bits per channel
		res_desc.res.linear.desc.w = 32; // bits per channel
	}
	else
	{
		return false;
	}
	return true;
}

cudaTextureObject_t nbody_engine_cuda::smemory::tex(size_t dev_n, evecsize vec_size)
{
	if(m_tex[dev_n] != 0)
	{
		return m_tex[dev_n];
	}

	cudaResourceDesc		res_desc;

	memset(&res_desc, 0, sizeof(res_desc));

	res_desc.resType = cudaResourceTypeLinear;
	res_desc.res.linear.devPtr = data(dev_n);
	res_desc.res.linear.sizeInBytes = size();

	if(!setup_texture_type<nbcoord_t>(res_desc, vec_size))
	{
		qDebug() << "Failed to create texture with vec_size =" << vec_size;
		return 0;
	}

	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(tex_desc));
	tex_desc.readMode = cudaReadModeElementType;
	tex_desc.addressMode[0] = cudaAddressModeClamp;
	tex_desc.filterMode = cudaFilterModePoint;
	tex_desc.normalizedCoords = 0;

	cudaCreateTextureObject(&m_tex[dev_n], &res_desc, &tex_desc, NULL);

	return m_tex[dev_n];
}

size_t nbody_engine_cuda::smemory::size() const
{
	return m_size;
}
