#include "nbody_engine_cuda_memory.h"

nbody_engine_cuda::smemory::smemory(size_t s) :
	m_data(NULL),
	m_size(0),
	m_tex(0)
{
	cudaError_t res = cudaMalloc(&m_data, s);
	if(res != cudaSuccess)
	{
		qDebug() << "Can't alloc " << s << "bytes. Err code" << res << cudaGetErrorString(res);
	}
	m_size = s;
}

nbody_engine_cuda::smemory::~smemory()
{
	if(m_tex != 0)
	{
		cudaDestroyTextureObject(m_tex);
	}
	cudaFree(m_data);
}

const void* nbody_engine_cuda::smemory::data() const
{
	return m_data;
}

void* nbody_engine_cuda::smemory::data()
{
	return m_data;
}

template<class T>
bool setup_texture_type(cudaResourceDesc&, int)
{
	return false;
}

template<>
bool setup_texture_type<float>(cudaResourceDesc& res_desc, int vec_size)
{
	res_desc.res.linear.desc.f = cudaChannelFormatKindFloat;
	if(vec_size == 1)
	{
		res_desc.res.linear.desc.x = 32; // bits per channel
	}
	else if(vec_size == 4)
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
bool setup_texture_type<double>(cudaResourceDesc& res_desc, int vec_size)
{
	res_desc.res.linear.desc.f = cudaChannelFormatKindSigned;
	if(vec_size == 1)
	{
		res_desc.res.linear.desc.x = 32; // bits per channel
		res_desc.res.linear.desc.y = 32; // bits per channel
	}
	else if(vec_size == 4)
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

cudaTextureObject_t nbody_engine_cuda::smemory::tex(int vec_size)
{
	if(m_tex != 0)
	{
		return m_tex;
	}

	cudaResourceDesc		res_desc;

	memset(&res_desc, 0, sizeof(res_desc));

	res_desc.resType = cudaResourceTypeLinear;
	res_desc.res.linear.devPtr = data();
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

	cudaCreateTextureObject(&m_tex, &res_desc, &tex_desc, NULL);

	return m_tex;
}

size_t nbody_engine_cuda::smemory::size() const
{
	return m_size;
}
