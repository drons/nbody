#ifndef NBODY_ENGINE_CUDA_MEMORY_H
#define NBODY_ENGINE_CUDA_MEMORY_H

#include "nbody_engine_cuda.h"
#include "nbody_engine_cuda_impl.h"

class nbody_engine_cuda::smemory : public nbody_engine::memory
{
	void*				m_data;
	size_t				m_size;
	cudaTextureObject_t	m_tex;
public:
	explicit smemory(size_t s);
	virtual ~smemory();
	void* data();
	const void* data() const;
	cudaTextureObject_t tex(int vec_size = 1);
	size_t size() const override;
};

#endif // NBODY_ENGINE_CUDA_MEMORY_H
