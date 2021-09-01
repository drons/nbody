#ifndef NBODY_ENGINE_OPENCL_H
#define NBODY_ENGINE_OPENCL_H

#include "nbody_engine.h"

class NBODY_DLL nbody_engine_opencl : public nbody_engine
{
	class	smemory;
	struct	data;
	data*	d;
	nbody_engine_opencl(const nbody_engine_opencl&) = delete;
	nbody_engine_opencl& operator = (const nbody_engine_opencl&) = delete;
public:
	nbody_engine_opencl();
	~nbody_engine_opencl();
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

	memory* create_buffer(size_t) override;
	void free_buffer(memory*) override;
	void read_buffer(void* dst, const memory* src) override;
	void write_buffer(memory* dst, const void* src) override;
	void copy_buffer(memory* a, const memory* b) override;
	void fill_buffer(memory* a, const nbcoord_t& value) override;
	void fmadd_inplace(memory* a, const memory* b, const nbcoord_t& c) override;
	void fmadd(memory* a, const memory* b, const memory* c, const nbcoord_t& d) override;
	void fmaddn_corr(memory* a, memory* corr, const memory_array& b,
					 const nbcoord_t* c, size_t csize) override;
	void fmaxabs(const memory* a, nbcoord_t& result) override;

	void print_info() const override;

	int select_devices(const QString& devices, bool verbose, bool prof);
	void set_block_size(int block_size);
protected:
	void fcompute_bh_impl(const nbcoord_t& t, const memory* y, memory* f,
						  nbcoord_t distance_to_node_radius_ratio,
						  bool cycle_traverse);
private:
	void synchronize_f(smemory* f);
};

#endif // NBODY_ENGINE_OPENCL_H
