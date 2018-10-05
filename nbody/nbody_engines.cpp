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
		//!todo compute device selection
		return new nbody_engine_opencl();
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
		nbcoord_t	distance_to_node_radius_ratio = param.value("distance_to_node_radius_ratio", 10).toDouble();
		nbody_engine_simple_bh::e_traverse_type	tt;
		if(strtt == "cycle")
		{
			tt = nbody_engine_simple_bh::ett_cycle;
		}
		else if(param.value("traverse_type") == "nested_tree")
		{
			tt = nbody_engine_simple_bh::ett_nested_tree;
		}
		else
		{
			qDebug() << "Invalid traverse_type. Allowed values are 'cycle' or 'nested_tree'";
			return NULL;
		}

		return new nbody_engine_simple_bh(distance_to_node_radius_ratio, tt);
	}

	return NULL;
}
