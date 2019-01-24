#include <QCoreApplication>
#include <QDebug>
#include <omp.h>
#include <iomanip>
#include <iostream>

#include "nbody_solvers.h"
#include "nbody_engines.h"
#include "nbody_data_stream.h"
#include "nbody_arg_parser.h"

int run(nbody_solver* solver, nbody_data* data, const QString& check_list, QVariantMap& bench_res, nbcoord_t max_time)
{
	nbcoord_t	dump_step = 0;
	nbcoord_t	check_step = max_time;

	data->set_check_list(check_list);

	qDebug() << "Solver:" << solver->type_name();
	solver->print_info();
	qDebug() << "Engine:" << solver->engine()->type_name();
	solver->engine()->print_info();
	data->print_statistics(solver->engine());
	solver->run(data, NULL, max_time, dump_step, check_step);
	bench_res["dP"] = data->get_impulce_err() / 100;
	bench_res["dL"] = data->get_impulce_moment_err() / 100;
	bench_res["dE"] = data->get_energy_err() / 100;
	bench_res["CC"] = static_cast<qulonglong>(solver->engine()->get_compute_count());
	return 0;
}

QVariantMap run(const QVariantMap& param, const QString& check_list, nbcoord_t max_time)
{
	nbody_data		data;
	nbcoord_t		box_size = 100;
	size_t			stars_count = param.value("stars_count", "1024").toUInt();

	data.make_universe(stars_count / 2, box_size, box_size, box_size);

	nbody_engine*	engine = nbody_create_engine(param);
	if(engine == NULL)
	{
		qDebug() << "Can't create engine" << param;
		return QVariantMap();
	}

	nbody_solver*	solver = nbody_create_solver(param);

	if(solver == NULL)
	{
		delete engine;
		qDebug() << "Can't create solver" << param;
		return QVariantMap();
	}

	nbcoord_t	max_step(solver->get_max_step());
	nbcoord_t	min_step(solver->get_min_step());
	if(min_step < 0)
	{
		solver->set_time_step(max_step, 0.1);
	}
	engine->init(&data);
	solver->set_engine(engine);
	QVariantMap	bench_res;

	double wtime = omp_get_wtime();
	int res = run(solver, &data, check_list, bench_res, max_time);
	wtime = omp_get_wtime() - wtime;

	if(res != 0)
	{
		return QVariantMap();
	}

	bench_res["time"] = wtime / engine->get_step();

	delete solver;
	delete engine;

	return bench_res;
}

void print_table_txt(const std::vector<QVariantMap>& params,
					 const std::vector<QVariant>& variable,
					 const std::vector<std::vector<QVariantMap>>& result,
					 const QString& param_header, const QStringList& result_field)
{
	using std::cout;
	{
		cout << std::setw(10);
		cout << std::setprecision(4);
		cout << std::setfill(' ');
		cout << param_header.toLocal8Bit().data() << " ";
		for(size_t j = 0; j < variable.size(); ++j)
		{
			cout << variable[j].toByteArray().data() << " ";
		}
		cout << std::endl;
	}
	for(size_t i = 0; i < result.size(); ++i)
	{
		cout << std::setw(10);
		cout << std::setprecision(4);
		cout << std::setfill(' ');
		cout << params[i][param_header].toByteArray().data();
		cout << " ";
		for(size_t j = 0; j < variable.size(); ++j)
		{
			if(result_field.size() == 1)
			{
				cout << result[i][j][result_field[0]].toDouble() << " ";
			}
			else
			{
				cout << "(";
				for(int r = 0; r != result_field.size(); ++r)
				{
					if(r != 0)
					{
						cout << ", ";
					}
					cout << result[i][j][result_field[r]].toDouble();
				}
				cout << ") ";
			}
		}
		cout << std::endl;
	}
}

void print_table_pgfplots(const std::vector<QVariantMap>& params,
						  const std::vector<QVariant>& variable,
						  const std::vector<std::vector<QVariantMap>>& result,
						  const QString& param_header, const QStringList& result_field)
{
	using std::cout;
	if(result_field.size() != 2)
	{
		qDebug() << "Result size must be 2";
		return;
	}
	std::vector<std::string>	plotcolors = {"red", "blue", "teal", "black"};
	std::vector<std::string>	plotmarks = {"*", "triangle*", "diamond*", "square*", "+", "x"};
	std::vector<std::string>	plotlines = {"solid", "dotted", "dashed"};
	std::vector<std::string>	styles;

	for(auto line : plotlines)
	{
		for(auto mark : plotmarks)
		{
			for(auto color : plotcolors)
			{
				styles.push_back(color + ",mark=" + mark + "," + line);
			}
		}
	}
	qDebug() << result_field;
	for(size_t i = 0; i < result.size(); ++i)
	{
		cout << std::setw(10);
		cout << std::setprecision(4);
		cout << "\\addplot [" << styles[i % result.size()] << "] coordinates { ";
		for(size_t j = 0; j < variable.size(); ++j)
		{
			cout << "(";
			for(int r = 0; r != result_field.size(); ++r)
			{
				if(r != 0)
				{
					cout << ", ";
				}
				cout << result[i][j][result_field[r]].toDouble();
			}
			cout << ") ";
		}
		cout << "};" << std::endl;
	}

	cout << "\\legend{";

	for(size_t i = 0; i < result.size(); ++i)
	{
		cout << "$" << params[i][param_header].toByteArray().data() << "$";
		if(i != result.size() - 1)
		{
			cout << ",";
		}
	}
	cout << "};" << std::endl;
}

void print_table(const std::vector<QVariantMap>& params,
				 const std::vector<QVariant>& variable,
				 const std::vector<std::vector<QVariantMap>>& result,
				 const QString& param_header, const QStringList& result_field,
				 const QString& format)
{
	if(format == "txt")
	{
		print_table_txt(params, variable, result, param_header, result_field);
	}
	else if(format == "pgfplots")
	{
		print_table_pgfplots(params, variable, result, param_header, result_field);
	}
	else
	{
		qDebug() << "Unknown table format" << format;
	}
}

int run_bench(const std::vector<QVariantMap>& params,
			  const std::vector<QVariant>& variable,
			  std::vector<std::vector<QVariantMap>>& result,
			  const QString& variable_field, const QString& check_list,
			  nbcoord_t max_time)
{
	for(size_t i = 0; i < params.size(); ++i)
	{
		for(size_t j = 0; j < variable.size(); ++j)
		{
			QVariantMap	p(params[i]);
			p[variable_field] = variable[j];
			result[i][j] = run(p, check_list, max_time);
			if(result[i][j].isEmpty())
			{
				qDebug() << "Bench failed with params" << p;
				return -1;
			}
		}
	}

	return 0;
}

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

	run_bench(params, stars_counts, result, variable_field, "PLV", 1);
	print_table(params, stars_counts, result, "engine", QStringList() << "time", format);
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

	run_bench(params, stars_counts, result, variable_field, "PLV", 1);
	print_table(params, stars_counts, result, "engine", QStringList() << "time", format);
}

void bench_cuda_tree(const QString& format)
{
	int		stars_count = 1024 * 256;
	QVariantMap param1(std::map<QString, QVariant>(
	{
		{"name", "dense"},
		{"engine", "cuda"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"max_step", 0.01}
	}));
	QVariantMap param2(std::map<QString, QVariant>(
	{
		{"name", "cuda_bh"},
		{"engine", "cuda_bh"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"max_step", 0.01}
	}));
	QVariantMap param3(std::map<QString, QVariant>(
	{
		{"name", "cuda_heap"},
		{"engine", "cuda_bh_tex"},
		{"tree_layout", "heap"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"max_step", 0.01}
	}));
	QVariantMap param4(std::map<QString, QVariant>(
	{
		{"name", "cuda_stackless"},
		{"engine", "cuda_bh_tex"},
		{"tree_layout", "heap_stackless"},
		{"solver", "euler"},
		{"stars_count", stars_count},
		{"max_step", 0.01}
	}));

	std::vector<QVariantMap>				params = {param1, param2, param3, param4};
	std::vector<QVariant>					block_sizes = {64, 128, 256, 512, 1024};
	QString									variable_field = "block_size";
	std::vector<std::vector<QVariantMap>>	result(params.size(), std::vector<QVariantMap>(block_sizes.size()));

	run_bench(params, block_sizes, result, variable_field, "PLV", 0.03);
	print_table(params, block_sizes, result, "name", QStringList() << "time", format);
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
		{"name", "rkf"},
		{"engine", engine},
		{"solver", "rkf"},
		{"stars_count", stars_count},
		{"min_step", "-1"}
	}));
	QVariantMap param09(std::map<QString, QVariant>(
	{
		{"name", "rkgl_1s"},
		{"engine", engine},
		{"solver", "rkgl"},
		{"refine_steps_count", 1},
		{"stars_count", stars_count}
	}));
	QVariantMap param10(std::map<QString, QVariant>(
	{
		{"name", "rkgl_3s"},
		{"engine", engine},
		{"solver", "rkgl"},
		{"refine_steps_count", 3},
		{"stars_count", stars_count}
	}));
	QVariantMap param11(std::map<QString, QVariant>(
	{
		{"name", "rklc_1s"},
		{"engine", engine},
		{"solver", "rklc"},
		{"refine_steps_count", 1},
		{"stars_count", stars_count},
		{"min_step", "-1"}
	}));
	QVariantMap param12(std::map<QString, QVariant>(
	{
		{"name", "rklc_3s"},
		{"engine", engine},
		{"solver", "rklc"},
		{"refine_steps_count", 3},
		{"stars_count", stars_count},
		{"min_step", "-1"}
	}));
	QVariantMap param13(std::map<QString, QVariant>(
	{
		{"name", "rklc_5s"},
		{"engine", engine},
		{"solver", "rklc"},
		{"refine_steps_count", 5},
		{"stars_count", stars_count},
		{"min_step", "-1"}
	}));
	QVariantMap param14(std::map<QString, QVariant>(
	{
		{"name", "trapeze"},
		{"engine", engine},
		{"solver", "trapeze"},
		{"stars_count", stars_count}
	}));

	std::vector<QVariantMap>				params = {param01, param02, param03, param04, param05, param06, param07, param08, param09, param10, param11, param12, param13, param14};
	std::vector<QVariant>					steps = {0.1, 0.1 / 8, 0.1 / (8 * 8), 0.1 / (8 * 8 * 8), 0.1 / (8 * 8 * 8 * 8), 0.1 / (8 * 8 * 8 * 8 * 8)};
	QString									variable_field = "max_step";
	std::vector<std::vector<QVariantMap>>	result(params.size(), std::vector<QVariantMap>(steps.size()));

	run_bench(params, steps, result, variable_field, "PLVE", 25);
	print_table(params, steps, result, "name", QStringList() << "CC" << "dE", format);
	print_table(params, steps, result, "name", QStringList() << "CC" << "dL", format);
	print_table(params, steps, result, "name", QStringList() << "CC" << "dP", format);
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
	else if(bench == "cuda_tree")
	{
		bench_cuda_tree(format);
	}
	else if(bench == "solver")
	{
		bench_solver(format);
	}

	return 0;
}
