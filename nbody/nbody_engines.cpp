#include "nbody_engines.h"


nbody_engine* nbody_create_engine(const QVariantMap& param)
{
	const QString type(param.value("engine").toString());

	if(type == "ah")
	{
		size_t		full_recompute_rate = param.value("full_recompute_rate", 1000).toUInt();
		nbcoord_t	max_dist = param.value("max_dist", 10).toDouble();
		nbcoord_t	min_force = param.value("min_force", 1e-4).toDouble();

		return new nbody_engine_ah(full_recompute_rate, max_dist, min_force);
	}
	else if(type == "block")
	{
		return new nbody_engine_block();
	}
#ifdef HAVE_CUDA
	else if(type == "cuda")
	{
		QString	devices(param.value("device", "0").toString());
		int		block_size(param.value("block_size", NBODY_DATA_BLOCK_SIZE).toInt());
		nbody_engine_cuda*	engine = new nbody_engine_cuda();

		if(0 != engine->select_devices(devices))
		{
			delete engine;
			return NULL;
		}
		engine->set_block_size(block_size);

		return engine;
	}
	else if(type == "cuda_bh")
	{
		int		block_size(param.value("block_size", NBODY_DATA_BLOCK_SIZE).toInt());
		nbcoord_t	distance_to_node_radius_ratio = param.value("distance_to_node_radius_ratio", 10).toDouble();
		nbody_engine_cuda_bh*	engine = new nbody_engine_cuda_bh(distance_to_node_radius_ratio);

		engine->set_block_size(block_size);

		return engine;
	}
	else if(type == "cuda_bh_tex")
	{
		int		block_size(param.value("block_size", NBODY_DATA_BLOCK_SIZE).toInt());
		nbcoord_t	distance_to_node_radius_ratio = param.value("distance_to_node_radius_ratio", 10).toDouble();
		QString		strtl(param.value("tree_layout", "heap").toString());
		e_tree_layout tl = tree_layout_from_str(strtl);

		if(tl != etl_heap && tl != etl_heap_stackless)
		{
			qDebug() << "Invalid tree_layout. Allowed values are 'heap' or 'heap_stackless'";
			return NULL;
		}
		nbody_engine_cuda_bh_tex*	engine = new nbody_engine_cuda_bh_tex(distance_to_node_radius_ratio, tl);

		engine->set_block_size(block_size);

		return engine;
	}
#endif //HAVE_CUDA
#ifdef HAVE_OPENCL
	else if(type == "opencl")
	{
		QString	devices(param.value("device", "0:0").toString());
		int		block_size(param.value("block_size", NBODY_DATA_BLOCK_SIZE).toInt());
		nbody_engine_opencl* engine = new nbody_engine_opencl();

		engine->set_block_size(block_size);

		if(0 != engine->select_devices(devices, param.value("verbose", "0").toInt() != 0,
									   param.value("oclprof", "0").toInt() != 0))
		{
			qDebug() << "Failed to select devices" << devices;
			delete engine;
			return NULL;
		}
		return engine;
	}
	else if(type == "opencl_bh")
	{
		QString	devices(param.value("device", "0:0").toString());
		int		block_size(param.value("block_size", NBODY_DATA_BLOCK_SIZE).toInt());
		QString		strtt(param.value("traverse_type", "cycle").toString());
		QString		strtl(param.value("tree_layout", "heap").toString());
		nbcoord_t	distance_to_node_radius_ratio = param.value("distance_to_node_radius_ratio", 10).toDouble();
		size_t		tree_build_rate = param.value("tree_build_rate", 0).toULongLong();
		e_tree_layout tl = tree_layout_from_str(strtl);
		e_traverse_type tt;

		if(tl != etl_heap && tl != etl_heap_stackless)
		{
			qDebug() << "Invalid tree_layout. Allowed values are 'heap' or 'heap_stackless'";
			return NULL;
		}
		if(strtt == "cycle")
		{
			tt = ett_cycle;
		}
		else if(strtt == "nested_tree")
		{
			tt = ett_nested_tree;
		}
		else
		{
			qDebug() << "Invalid traverse_type. Allowed values are 'cycle' or 'nested_tree'";
			return NULL;
		}

		nbody_engine_opencl_bh* engine = new nbody_engine_opencl_bh(distance_to_node_radius_ratio,
																	tt, tree_build_rate, tl);

		engine->set_block_size(block_size);
		if(0 != engine->select_devices(devices, param.value("verbose", "0").toInt() != 0,
									   param.value("oclprof", "0").toInt() != 0))
		{
			qDebug() << "Failed to select devices" << devices;
			delete engine;
			return NULL;
		}
		return engine;
	}
#endif
	else if(type == "openmp")
	{
		return new nbody_engine_openmp();
	}
	else if(type == "simple")
	{
		return new nbody_engine_simple();
	}
	else if(type == "simple_bh")
	{
		QString		strtt(param.value("traverse_type", "cycle").toString());
		QString		strtl(param.value("tree_layout", "tree").toString());
		nbcoord_t	distance_to_node_radius_ratio = param.value("distance_to_node_radius_ratio", 10).toDouble();
		size_t		tree_build_rate = param.value("tree_build_rate", 0).toULongLong();
		e_traverse_type	tt;

		if(strtt == "cycle")
		{
			tt = ett_cycle;
		}
		else if(strtt == "nested_tree")
		{
			tt = ett_nested_tree;
		}
		else
		{
			qDebug() << "Invalid traverse_type. Allowed values are 'cycle' or 'nested_tree'";
			return NULL;
		}

		e_tree_layout tl = tree_layout_from_str(strtl);
		switch(tl)
		{
		case etl_unknown:
			qDebug() << "Invalid tree_layout. Allowed values are 'tree', 'heap' or 'heap_stackless'";
			return nullptr;
		case etl_tree:
			return new nbody_engine_simple_bh_tree(distance_to_node_radius_ratio, tt, tree_build_rate);
		case etl_heap:
			return new nbody_engine_simple_bh_heap(distance_to_node_radius_ratio, tt, tree_build_rate);
		case etl_heap_stackless:
			return new nbody_engine_simple_bh_heap_stackless(distance_to_node_radius_ratio, tt, tree_build_rate);
		}
	}

	return NULL;
}
