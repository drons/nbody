#include <QCoreApplication>
#include <QDebug>
#include <omp.h>

#include "bench.h"
#include "nbody_data_stream.h"
#include "nbody_arg_parser.h"

void bench_cpu(const QString& format)
{
	QVariantMap param1(std::map<QString, QVariant>(
	{
		{"engine", "simple"},
		{"solver", "euler"}
	}));
	QVariantMap param2(std::map<QString, QVariant>(
	{
		{"engine", "openmp"},
		{"solver", "euler"}
	}));
	QVariantMap param3(std::map<QString, QVariant>(
	{
		{"engine", "block"},
		{"solver", "euler"}
	}));

	std::vector<QVariantMap>				params = {param1, param2, param3};
	std::vector<QVariant>					stars_counts = {1024, 2048, 4096, 8192};
	QString									variable_field = "stars_count";
	std::vector<std::vector<QVariantMap>>	result(params.size(), std::vector<QVariantMap>(stars_counts.size()));

	run_bench(params, stars_counts, result, variable_field, QString(), 1);
	print_table(params, stars_counts, result, "engine", QStringList() << "time", QStringList(), format);
}

void bench_gpu(const QString& format)
{
	QVariantMap param1(std::map<QString, QVariant>(
	{
		{"engine", "opencl"},
		{"solver", "euler"}
	}));
	QVariantMap param2(std::map<QString, QVariant>(
	{
		{"engine", "opencl_bh"},
		{"solver", "euler"}
	}));
	QVariantMap param3(std::map<QString, QVariant>(
	{
		{"engine", "cuda"},
		{"solver", "euler"}
	}));
	QVariantMap param4(std::map<QString, QVariant>(
	{
		{"engine", "cuda_bh"},
		{"solver", "euler"}
	}));
	std::vector<QVariantMap>				params = {param1, param2, param3, param4};
	std::vector<QVariant>					stars_counts = {1024 * 8, 1024 * 16, 1024 * 32, 1024 * 64, 1024 * 128, 1024 * 256 };
	QString									variable_field = "stars_count";
	std::vector<std::vector<QVariantMap>>	result(params.size(), std::vector<QVariantMap>(stars_counts.size()));

	run_bench(params, stars_counts, result, variable_field, QString(), 1);
	print_table(params, stars_counts, result, "engine", QStringList() << "time", QStringList(), format);
}

void bench_gpu_tree_block(const QString& format, const QVariantMap& param)
{
	int		stars_count = param.value("stars_count", 131072).toInt();
	double	distance_to_node_radius_ratio = param.value("distance_to_node_radius_ratio", 10).toDouble();
	QVariantMap param1(std::map<QString, QVariant>(
	{
		{"name", "opencl+dense"},
		{"engine", "opencl"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"distance_to_node_radius_ratio", distance_to_node_radius_ratio},
		{"max_step", 0.01}
	}));
	QVariantMap param2(std::map<QString, QVariant>(
	{
		{"name", "cuda+dense"},
		{"engine", "cuda"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"distance_to_node_radius_ratio", distance_to_node_radius_ratio},
		{"max_step", 0.01}
	}));
	QVariantMap param3(std::map<QString, QVariant>(
	{
		{"name", "opencl+heap+cycle"},
		{"engine", "opencl_bh"},
		{"traverse_type", "cycle"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"distance_to_node_radius_ratio", distance_to_node_radius_ratio},
		{"max_step", 0.01}
	}));
	QVariantMap param4(std::map<QString, QVariant>(
	{
		{"name", "opencl+heap+nested"},
		{"engine", "opencl_bh"},
		{"traverse_type", "nested_tree"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"distance_to_node_radius_ratio", distance_to_node_radius_ratio},
		{"max_step", 0.01}
	}));
	QVariantMap param5(std::map<QString, QVariant>(
	{
		{"name", "cuda+heap+nested"},
		{"engine", "cuda_bh"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"distance_to_node_radius_ratio", distance_to_node_radius_ratio},
		{"max_step", 0.01}
	}));
	QVariantMap param6(std::map<QString, QVariant>(
	{
		{"name", "cuda+heap+nested+tex"},
		{"engine", "cuda_bh_tex"},
		{"tree_layout", "heap"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"distance_to_node_radius_ratio", distance_to_node_radius_ratio},
		{"max_step", 0.01}
	}));
	QVariantMap param7(std::map<QString, QVariant>(
	{
		{"name", "cuda+heap+nested+tex+stackless"},
		{"engine", "cuda_bh_tex"},
		{"tree_layout", "heap_stackless"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"distance_to_node_radius_ratio", distance_to_node_radius_ratio},
		{"max_step", 0.01}
	}));

	std::vector<QVariantMap>				params = {param1, param2, param3, param4, param5, param6, param7};
	std::vector<QVariant>					block_sizes = {8, 16, 32, 64, 128, 256, 512, 1024};
	QString									variable_field = "block_size";
	std::vector<std::vector<QVariantMap>>	result(params.size(), std::vector<QVariantMap>(block_sizes.size()));

	run_bench(params, block_sizes, result, variable_field, QString(), 0.1);
	print_table(params, block_sizes, result, "name", QStringList() << "time", QStringList(), format);
}

void bench_gpu_tree_ratio(const QString& format, const QVariantMap& param)
{
	int		stars_count = param.value("stars_count", 131072).toInt();
	size_t		ratio_count = param.value("ratio_count", 0).toUInt();
	QVariantMap param1(std::map<QString, QVariant>(
	{
		{"name", "opencl+heap+cycle"},
		{"engine", "opencl_bh"},
		{"traverse_type", "cycle"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"block_size", 8},
		{"max_step", 0.01}
	}));
	QVariantMap param2(std::map<QString, QVariant>(
	{
		{"name", "opencl+heap+nested"},
		{"engine", "opencl_bh"},
		{"traverse_type", "nested_tree"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"block_size", 32},
		{"max_step", 0.01}
	}));
	QVariantMap param3(std::map<QString, QVariant>(
	{
		{"name", "cuda+heap+nested"},
		{"engine", "cuda_bh"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"block_size", 32},
		{"max_step", 0.01}
	}));
	QVariantMap param4(std::map<QString, QVariant>(
	{
		{"name", "cuda+heap+nested+tex"},
		{"engine", "cuda_bh_tex"},
		{"tree_layout", "heap"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"block_size", 64},
		{"max_step", 0.01}
	}));
	QVariantMap param5(std::map<QString, QVariant>(
	{
		{"name", "cuda+heap+nested+tex+stackless"},
		{"engine", "cuda_bh_tex"},
		{"tree_layout", "heap_stackless"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"block_size", 256},
		{"max_step", 0.01}
	}));

	std::vector<QVariantMap>				params = {param1, param2, param3, param4, param5};
	std::vector<QVariant>					ratio = {0.1, 0.5, 1, 2, 4, 16, 64, 256, 1024};

	if(ratio_count > 0 && ratio_count < ratio.size())
	{
		ratio = std::vector<QVariant>(ratio.begin(), ratio.begin() + ratio_count);
	}

	QString									variable_field = "distance_to_node_radius_ratio";
	std::vector<std::vector<QVariantMap>>	result(params.size(), std::vector<QVariantMap>(ratio.size()));

	run_bench(params, ratio, result, variable_field, QString(), 0.1);
	print_table(params, ratio, result, "name", QStringList() << "distance_to_node_radius_ratio" << "time",
				QStringList() << "$\\lambda_{crit}$" << "Step time (s)", format);
}

void bench_solver(const QString& format)
{
	int		stars_count = 512;
	QString	engine("block");
	QVariantMap param01(std::map<QString, QVariant>(
	{
		{"name", "adams2"},
		{"engine", engine},
		{"solver", "adams"},
		{"rank", 2},
		{"starter_solver", "rkdp"},
		{"stars_count", stars_count}
	}));
	QVariantMap param02(std::map<QString, QVariant>(
	{
		{"name", "adams3"},
		{"engine", engine},
		{"solver", "adams"},
		{"rank", 3},
		{"starter_solver", "rkdp"},
		{"stars_count", stars_count}
	}));
	QVariantMap param03(std::map<QString, QVariant>(
	{
		{"name", "adams5"},
		{"engine", engine},
		{"solver", "adams"},
		{"rank", 5},
		{"starter_solver", "rkdp"},
		{"stars_count", stars_count}
	}));
	QVariantMap param04(std::map<QString, QVariant>(
	{
		{"name", "euler"},
		{"engine", engine},
		{"solver", "euler"},
		{"stars_count", stars_count}
	}));
	QVariantMap param05(std::map<QString, QVariant>(
	{
		{"name", "rk4"},
		{"engine", engine},
		{"solver", "rk4"},
		{"stars_count", stars_count}
	}));
	QVariantMap param06(std::map<QString, QVariant>(
	{
		{"name", "rkck"},
		{"engine", engine},
		{"solver", "rkck"},
		{"stars_count", stars_count},
		{"min_step", "-1"}
	}));
	QVariantMap param07(std::map<QString, QVariant>(
	{
		{"name", "rkdp"},
		{"engine", engine},
		{"solver", "rkdp"},
		{"stars_count", stars_count},
		{"min_step", "-1"}
	}));
	QVariantMap param08(std::map<QString, QVariant>(
	{
		{"name", "rkdverk"},
		{"engine", engine},
		{"solver", "rkdverk"},
		{"stars_count", stars_count},
		{"min_step", "-1"}
	}));
	QVariantMap param09(std::map<QString, QVariant>(
	{
		{"name", "rkf"},
		{"engine", engine},
		{"solver", "rkf"},
		{"stars_count", stars_count},
		{"min_step", "-1"}
	}));
	QVariantMap param10(std::map<QString, QVariant>(
	{
		{"name", "rkgl_1s"},
		{"engine", engine},
		{"solver", "rkgl"},
		{"refine_steps_count", 1},
		{"stars_count", stars_count}
	}));
	QVariantMap param11(std::map<QString, QVariant>(
	{
		{"name", "rkgl_3s"},
		{"engine", engine},
		{"solver", "rkgl"},
		{"refine_steps_count", 3},
		{"stars_count", stars_count}
	}));
	QVariantMap param12(std::map<QString, QVariant>(
	{
		{"name", "rklc_1s"},
		{"engine", engine},
		{"solver", "rklc"},
		{"refine_steps_count", 1},
		{"stars_count", stars_count},
		{"min_step", "-1"}
	}));
	QVariantMap param13(std::map<QString, QVariant>(
	{
		{"name", "rklc_3s"},
		{"engine", engine},
		{"solver", "rklc"},
		{"refine_steps_count", 3},
		{"stars_count", stars_count},
		{"min_step", "-1"}
	}));
	QVariantMap param14(std::map<QString, QVariant>(
	{
		{"name", "rklc_5s"},
		{"engine", engine},
		{"solver", "rklc"},
		{"refine_steps_count", 5},
		{"stars_count", stars_count},
		{"min_step", "-1"}
	}));
	QVariantMap param15(std::map<QString, QVariant>(
	{
		{"name", "trapeze"},
		{"engine", engine},
		{"solver", "trapeze"},
		{"stars_count", stars_count}
	}));

	std::vector<QVariantMap>				params = {param01, param02, param03, param04, param05, param06, param07, param08, param09, param10, param11, param12, param13, param14, param15};
	std::vector<QVariant>					steps = {0.1, 0.1 / 8, 0.1 / (8 * 8), 0.1 / (8 * 8 * 8), 0.1 / (8 * 8 * 8 * 8), 0.1 / (8 * 8 * 8 * 8 * 8)};
	QString									variable_field = "max_step";
	std::vector<std::vector<QVariantMap>>	result(params.size(), std::vector<QVariantMap>(steps.size()));

	run_bench(params, steps, result, variable_field, "PLVE", 2.5);
	print_table(params, steps, result, "name", QStringList() << "CC" << "dE",
				QStringList() << "$f_n$ compute count" << "$dE/E_0$", format);
	print_table(params, steps, result, "name", QStringList() << "CC" << "dL",
				QStringList() << "$f_n$ compute count" << "$dL/L_0$", format);
	print_table(params, steps, result, "name", QStringList() << "CC" << "dP",
				QStringList() << "$f_n$ compute count" << "$dP/P_0$", format);
}

void bench_solver_quad(const QString& format)
{
	int		stars_count = 512;
	double	error_threshold = 1e-4;
	QString	engine("block");

	QVariantMap param01(std::map<QString, QVariant>(
	{
		{"name", "adams5"},
		{"engine", engine},
		{"solver", "adams"},
		{"rank", 5},
		{"starter_solver", "rkdp"},
		{"error_threshold", error_threshold},
		{"stars_count", stars_count}
	}));
	QVariantMap param02(std::map<QString, QVariant>(
	{
		{"name", "euler"},
		{"engine", engine},
		{"solver", "euler"},
		{"stars_count", stars_count}
	}));
	QVariantMap param03(std::map<QString, QVariant>(
	{
		{"name", "rk4"},
		{"engine", engine},
		{"solver", "rk4"},
		{"stars_count", stars_count}
	}));
	QVariantMap param04(std::map<QString, QVariant>(
	{
		{"name", "rkck"},
		{"engine", engine},
		{"solver", "rkck"},
		{"error_threshold", error_threshold},
		{"stars_count", stars_count},
		{"min_step", "-1"}
	}));
	QVariantMap param05(std::map<QString, QVariant>(
	{
		{"name", "rkdp"},
		{"engine", engine},
		{"solver", "rkdp"},
		{"error_threshold", error_threshold},
		{"stars_count", stars_count},
		{"min_step", "-1"}
	}));
	QVariantMap param06(std::map<QString, QVariant>(
	{
		{"name", "rkdverk"},
		{"engine", engine},
		{"solver", "rkdverk"},
		{"error_threshold", error_threshold},
		{"stars_count", stars_count},
		{"min_step", "-1"}
	}));
	std::vector<QVariantMap>				params = {param01, param02, param03, param04, param05, param06};
	std::vector<QVariant>					steps = {0.1, 0.1 / 8, 0.1 / (8 * 8), 0.1 / (8 * 8 * 8), 0.1 / (8 * 8 * 8 * 8), 0.1 / (8 * 8 * 8 * 8 * 8)};
	QString									variable_field = "max_step";
	std::vector<std::vector<QVariantMap>>	result(params.size(), std::vector<QVariantMap>(steps.size()));

	run_bench(params, steps, result, variable_field, "PLVE", 2.5);
	print_table(params, steps, result, "name", QStringList() << "CC" << "dE",
				QStringList() << "$f_n$ compute count" << "$dE/E_0$", format);
	print_table(params, steps, result, "name", QStringList() << "CC" << "dL",
				QStringList() << "$f_n$ compute count" << "$dL/L_0$", format);
	print_table(params, steps, result, "name", QStringList() << "CC" << "dP",
				QStringList() << "$f_n$ compute count" << "$dP/P_0$", format);
}


void bench_cpu_tree(const QString& format)
{
	int		stars_count = 1024 * 32;

	QVariantMap param01(std::map<QString, QVariant>(
	{
		{"name", "cycle+tree"},
		{"engine", "simple_bh"},
		{"traverse_type", "cycle"},
		{"tree_layout", "tree"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"max_step", "0.01"}
	}));
	QVariantMap param02(std::map<QString, QVariant>(
	{
		{"name", "cycle+heap"},
		{"engine", "simple_bh"},
		{"traverse_type", "cycle"},
		{"tree_layout", "heap"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"max_step", "0.01"}
	}));
	QVariantMap param03(std::map<QString, QVariant>(
	{
		{"name", "cycle+heap stackless"},
		{"engine", "simple_bh"},
		{"traverse_type", "cycle"},
		{"tree_layout", "heap_stackless"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"max_step", "0.01"}
	}));
	QVariantMap param04(std::map<QString, QVariant>(
	{
		{"name", "nested tree+tree"},
		{"engine", "simple_bh"},
		{"traverse_type", "nested_tree"},
		{"tree_layout", "tree"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"max_step", "0.01"}
	}));
	QVariantMap param05(std::map<QString, QVariant>(
	{
		{"name", "nested tree+heap"},
		{"engine", "simple_bh"},
		{"traverse_type", "nested_tree"},
		{"tree_layout", "heap"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"max_step", "0.01"}
	}));
	QVariantMap param06(std::map<QString, QVariant>(
	{
		{"name", "nested tree+heap stackless"},
		{"engine", "simple_bh"},
		{"traverse_type", "nested_tree"},
		{"tree_layout", "heap_stackless"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"max_step", "0.01"}
	}));
	QVariantMap param07(std::map<QString, QVariant>(
	{
		{"name", "openmp+block+optimization"},
		{"engine", "block"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"max_step", "0.01"}
	}));
	std::vector<QVariantMap>				params = {param01, param02, param03, param04, param05, param06, param07};
	std::vector<QVariant>					ratio = {0.1, 0.5, 1, 2, 4, 16, 64, 256, 1024};
	QString									variable_field = "distance_to_node_radius_ratio";
	std::vector<std::vector<QVariantMap>>	result(params.size(), std::vector<QVariantMap>(ratio.size()));

	run_bench(params, ratio, result, variable_field, "PLVE", 1);
	print_table(params, ratio, result, "name", QStringList() << "distance_to_node_radius_ratio" << "dE",
				QStringList() << "$\\lambda_{crit}$" << "$dE/E_0$", format);
	run_bench(params, ratio, result, variable_field, QString(), 1);
	print_table(params, ratio, result, "name", QStringList() << "distance_to_node_radius_ratio" << "time",
				QStringList() << "$\\lambda_{crit}$" << "Step time (s)", format);
}

int main(int argc, char* argv[])
{
	QCoreApplication	a(argc, argv);
	QVariantMap			param(nbody_parse_arguments(argc, argv));
	const QString		bench(param.value("bench", "cpu").toString());
	const QString		format(param.value("format", "txt").toString());

	if(bench == "cpu")
	{
		bench_cpu(format);
	}
	else if(bench == "gpu")
	{
		bench_gpu(format);
	}
	else if(bench == "gpu_tree_block")
	{
		bench_gpu_tree_block(format, param);
	}
	else if(bench == "gpu_tree_ratio")
	{
		bench_gpu_tree_ratio(format, param);
	}
	else if(bench == "solver")
	{
		bench_solver(format);
	}
	else if(bench == "solver_quad")
	{
		bench_solver_quad(format);
	}
	else if(bench == "cpu_tree")
	{
		bench_cpu_tree(format);
	}

	return 0;
}
