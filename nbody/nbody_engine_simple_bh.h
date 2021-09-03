#ifndef NBODY_ENGINE_SIMPLE_BH_H
#define NBODY_ENGINE_SIMPLE_BH_H

#include "nbody_engine_openmp.h"

static constexpr	size_t SPACE_DIMENSIONS = 3;
static constexpr	size_t DIM_NUM_X = 0;
static constexpr	size_t DIM_NUM_Y = 1;
static constexpr	size_t DIM_NUM_Z = 2;
static constexpr	size_t MAX_STACK_SIZE = 64;

enum e_traverse_type
{
	ett_cycle,
	ett_nested_tree
};
enum e_tree_layout
{
	etl_tree,
	etl_heap,
	etl_heap_stackless,

	etl_unknown = 0xffffffff
};

const char NBODY_DLL* tree_layout_name(e_tree_layout tree_layout);
e_tree_layout NBODY_DLL tree_layout_from_str(const QString& name);

/*!
  Barnes–Hut simulation
  https://en.wikipedia.org/wiki/Barnes%E2%80%93Hut_simulation
 */
class nbody_engine_simple_bh : public nbody_engine_openmp
{
private:
	nbcoord_t			m_distance_to_node_radius_ratio;
	e_traverse_type		m_traverse_type;
	e_tree_layout		m_tree_layout;
protected:
	nbody_engine_simple_bh(nbcoord_t distance_to_node_radius_ratio,
						   e_traverse_type tt, e_tree_layout tl);
public:
	void print_info() const override;
protected:
	template<class T>
	void space_subdivided_fcompute(const smemory* y, smemory* f);
};

//Tree layout
class NBODY_DLL nbody_engine_simple_bh_tree : public nbody_engine_simple_bh
{
public:
	nbody_engine_simple_bh_tree(nbcoord_t distance_to_node_radius_ratio = 0,
								e_traverse_type tt = ett_cycle);
	const char* type_name() const override;
	void fcompute(const nbcoord_t& t, const memory* y, memory* f) override;
};

//Heap layout
class NBODY_DLL nbody_engine_simple_bh_heap : public nbody_engine_simple_bh
{
public:
	nbody_engine_simple_bh_heap(nbcoord_t distance_to_node_radius_ratio = 0,
								e_traverse_type tt = ett_cycle);
	const char* type_name() const override;
	void fcompute(const nbcoord_t& t, const memory* y, memory* f) override;
};

//Heap-stackless layout
class NBODY_DLL nbody_engine_simple_bh_heap_stackless : public nbody_engine_simple_bh
{
public:
	nbody_engine_simple_bh_heap_stackless(nbcoord_t distance_to_node_radius_ratio = 0,
										  e_traverse_type tt = ett_cycle);
	const char* type_name() const override;
	void fcompute(const nbcoord_t& t, const memory* y, memory* f) override;
};
#endif // NBODY_ENGINE_SIMPLE_BH_H
