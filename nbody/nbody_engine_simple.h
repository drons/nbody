#ifndef NBODY_ENGINE_SIMPLE_H
#define NBODY_ENGINE_SIMPLE_H

#include "nbody_engine.h"

class NBODY_DLL nbody_engine_simple : public nbody_engine
{
protected:
	class smemory : public memory
	{
		void*	m_data;
		size_t	m_size;
	public:
		explicit smemory(size_t);
		~smemory();
		void* data();
		const void* data() const;
		size_t size() const override;
	};

	smemory*			m_mass;
	smemory*			m_y;
	nbody_data*			m_data;
public:
	nbody_engine_simple();
	~nbody_engine_simple();
	const char* type_name() const override;
	void init(nbody_data* data) override;
	void get_data(nbody_data* data) override;
	size_t problem_size() const override;
	memory* get_y() override;
	void advise_time(const nbcoord_t& dt) override;
	nbcoord_t get_time() const override;
	void set_time(nbcoord_t t) override;
	size_t get_step() const override;
	void set_step(size_t s) override;
	void fcompute(const nbcoord_t& t, const memory* y, memory* f) override;

	smemory* create_buffer(size_t) override;
	void free_buffer(memory*) override;
	void read_buffer(void* dst, const memory* src) override;
	void write_buffer(memory* dst, const void* src) override;
	void copy_buffer(memory* a, const memory* b) override;
	void fill_buffer(memory* a, const nbcoord_t& value) override;
	void fmadd_inplace(memory* a, const memory* b, const nbcoord_t& c) override;
	void fmadd(memory* a, const memory* b, const memory* c, const nbcoord_t& d) override;
	void fmaddn_corr(memory* a, memory* corr, const memory_array& b, const nbcoord_t* c) override;
	void fmaxabs(const memory* a, nbcoord_t& result) override;
};

#endif // NBODY_ENGINE_SIMPLE_H
