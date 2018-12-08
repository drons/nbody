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
		nbcoord_t	distance_to_node_radius_ratio = param.value("distance_to_node_radius_ratio", 10).toDouble();
		nbody_engine_opencl_bh* engine = new nbody_engine_opencl_bh(distance_to_node_radius_ratio);

		if(strtt != "cycle" && strtt != "nested_tree")
		{
			qDebug() << "Invalid traverse_type. Allowed values are 'cycle' or 'nested_tree'";
			return NULL;
		}

		engine->set_block_size(block_size);
		engine->set_cycle_traverse(strtt == "cycle");

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
		nbody_engine_simple_bh::e_traverse_type	tt;
		nbody_engine_simple_bh::e_tree_layout	tl;

		if(strtt == "cycle")
		{
			tt = nbody_engine_simple_bh::ett_cycle;
		}
		else if(strtt == "nested_tree")
		{
			tt = nbody_engine_simple_bh::ett_nested_tree;
		}
		else
		{
			qDebug() << "Invalid traverse_type. Allowed values are 'cycle' or 'nested_tree'";
			return NULL;
		}

		if(strtl == "tree")
		{
			tl = nbody_engine_simple_bh::etl_tree;
		}
		else if(strtl == "heap")
		{
			tl = nbody_engine_simple_bh::etl_heap;
		}
		else
		{
			qDebug() << "Invalid tree_layout. Allowed values are 'tree' or 'heap'";
			return NULL;
		}

		return new nbody_engine_simple_bh(distance_to_node_radius_ratio, tt, tl);
	}

	return NULL;
}
