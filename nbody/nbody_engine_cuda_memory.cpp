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

cudaTextureObject_t nbody_engine_cuda::smemory::tex()
{
	if(m_tex != 0)
	{
		return m_tex;
	}

	cudaResourceDesc		res_desc;
	cudaResourceViewDesc	view_desc;

	memset(&res_desc, 0, sizeof(res_desc));
	memset(&view_desc, 0, sizeof(view_desc));

	res_desc.resType = cudaResourceTypeLinear;
	res_desc.res.linear.devPtr = data();
	if(sizeof(nbcoord_t) == sizeof(float))
	{
		res_desc.res.linear.desc.f = cudaChannelFormatKindFloat;
		res_desc.res.linear.desc.x = 32; // bits per channel
		view_desc.format = cudaResViewFormatFloat1;
	}
	else if(sizeof(nbcoord_t) == sizeof(double))
	{
		res_desc.res.linear.desc.f = cudaChannelFormatKindSigned;
		res_desc.res.linear.desc.x = 32; // bits per channel
		view_desc.format = cudaResViewFormatSignedInt2;
	}

	res_desc.res.linear.sizeInBytes = size();
	view_desc.width = size() / sizeof(nbcoord_t);
	view_desc.height = 1;
	view_desc.depth = 1;

	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(tex_desc));
	tex_desc.readMode = cudaReadModeElementType;
	tex_desc.addressMode[0] = cudaAddressModeClamp;
	tex_desc.filterMode = cudaFilterModePoint;

	cudaCreateTextureObject(&m_tex, &res_desc, &tex_desc, NULL);

	return m_tex;
}

size_t nbody_engine_cuda::smemory::size() const
{
	return m_size;
}
