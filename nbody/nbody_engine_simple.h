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
		virtual ~smemory();
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
	virtual const char* type_name() const override;
	virtual void init(nbody_data* data) override;
	virtual void get_data(nbody_data* data) override;
	virtual size_t problem_size() const override;
	virtual memory* get_y() override;
	virtual void advise_time(const nbcoord_t& dt) override;
	virtual nbcoord_t get_time() const override;
	virtual void set_time(nbcoord_t t) override;
	virtual size_t get_step() const override;
	virtual void set_step(size_t s) override;
	virtual void fcompute(const nbcoord_t& t, const memory* y, memory* f) override;

	virtual smemory* create_buffer(size_t) override;
	virtual void free_buffer(memory*) override;
	virtual void read_buffer(void* dst, const memory* src) override;
	virtual void write_buffer(memory* dst, const void* src) override;
	virtual void copy_buffer(memory* a, const memory* b) override;
	virtual void fill_buffer(memory* a, const nbcoord_t& value) override;
	virtual void fmadd_inplace(memory* a, const memory* b, const nbcoord_t& c) override;
	virtual void fmadd(memory* a, const memory* b, const memory* c, const nbcoord_t& d) override;
	virtual void fmaxabs(const memory* a, nbcoord_t& result) override;
};

#endif // NBODY_ENGINE_SIMPLE_H
