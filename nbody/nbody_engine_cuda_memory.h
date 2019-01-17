#ifndef NBODY_ENGINE_CUDA_MEMORY_H
#define NBODY_ENGINE_CUDA_MEMORY_H

#include "nbody_engine_cuda.h"
#include "nbody_engine_cuda_impl.h"

class nbody_engine_cuda::smemory : public nbody_engine::memory
{
	std::vector<void*>					m_data;
	std::vector<cudaTextureObject_t>	m_tex;
	size_t								m_size;
	std::vector<int>					m_device_ids;
public:
	enum evecsize {evs1 = 1, evs4 = 4};
	explicit smemory(size_t size, const std::vector<int>& dev_ids);
	virtual ~smemory();
	void* data(size_t dev_n);
	const void* data(size_t dev_id) const;
	cudaTextureObject_t tex(size_t dev_n, evecsize vec_size = evs1);
	size_t size() const override;
};

#endif // NBODY_ENGINE_CUDA_MEMORY_H
