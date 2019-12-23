#ifndef NBODY_BENCH_H
#define NBODY_BENCH_H

#include <iomanip>
#include <iostream>

#include "nbody_engines.h"
#include "nbody_solvers.h"

static std::pair<nbcoord_t, nbcoord_t>
compare_data(const nbvertex_t* a,
			 const nbvertex_t* b,
			 size_t count)
{
	nbcoord_t	total = 0_f;
	nbcoord_t	max_diff = 0_f;
	for(size_t n = 0; n < count; ++n)
	{
		nbcoord_t	diff((a[n] - b[n]).length());
		total += diff;
		max_diff = std::max(max_diff, diff);
	}
	return std::make_pair(total / count, max_diff);
}

static int run(nbody_solver* solver,
			   nbody_data* data,
			   const QString& check_list,
			   QVariantMap& bench_res,
			   nbcoord_t max_time,
			   nbcoord_t check_step)
{
	nbcoord_t	dump_step = 0;

	data->set_check_list(check_list);

	qDebug() << "Solver:" << solver->type_name() << max_time << check_step;
	solver->print_info();
	qDebug() << "Engine:" << solver->engine()->type_name();
	solver->engine()->print_info();
	data->print_statistics(solver->engine());
	solver->run(data, NULL, max_time, dump_step, check_step);
	bench_res["dP"] = static_cast<double>(data->get_impulce_err() / 100);
	bench_res["dL"] = static_cast<double>(data->get_impulce_moment_err() / 100);
	bench_res["dE"] = static_cast<double>(data->get_energy_err() / 100);
	bench_res["CC"] = static_cast<qulonglong>(solver->engine()->get_compute_count());
	return 0;
}

QVariantMap run(const QVariantMap& param,
				const QString& check_list,
				nbcoord_t max_time,
				std::shared_ptr<nbody_step_visitor> step_visitor = nullptr)
{
	nbody_data	data;
	QString		initial_state(param.value("initial_state", QString()).toString());

	if(initial_state.isEmpty())
	{
		size_t		stars_count = param.value("stars_count", "1024").toUInt();
		nbcoord_t	box_size = 100;
		data.make_universe(stars_count / 2, box_size, box_size, box_size);
	}
	else
	{
		QString	initial_state_type(param.value("initial_type", "ADK").toString());
		if(!data.load_initial(initial_state, initial_state_type))
		{
			qDebug() << "Can't load initial state" << initial_state;
			return QVariantMap();
		}
	}

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
	nbcoord_t	check_step =
		param.value("check_step", static_cast<double>(max_time)).toDouble();
	if(min_step < 0)
	{
		solver->set_time_step(max_step, max_step);
	}
	engine->init(&data);
	solver->set_engine(engine);
	QVariantMap	bench_res(param);
	if(step_visitor != nullptr)
	{
		solver->add_check_visitor(step_visitor);
	}
	double wtime = omp_get_wtime();
	int res = run(solver, &data, check_list, bench_res, max_time, check_step);
	wtime = omp_get_wtime() - wtime;

	if(res != 0)
	{
		qDebug() << "Solver run failed";
		return QVariantMap();
	}

	bench_res["time"] = wtime / engine->get_step();

	delete solver;
	delete engine;

	const QString	end_state(param.value("end_state", QString()).toString());
	if(!end_state.isEmpty())
	{
		data.save(end_state);
	}
	const QString	expected_state(param.value("expected_state", QString()).toString());
	if(!expected_state.isEmpty())
	{
		nbody_data	expected_data;
		expected_data.load_initial(expected_state, "G1");
		size_t	count = data.get_count();
		if(expected_data.get_count() == count)
		{
			const auto dr = compare_data(data.get_vertites(), expected_data.get_vertites(), count);
			const auto dv = compare_data(data.get_velosites(), expected_data.get_velosites(), count);
			bench_res["dR"] = static_cast<double>(dr.first);
			bench_res["dR_max"] = static_cast<double>(dr.second);
			bench_res["dV"] = static_cast<double>(dv.first);
			bench_res["dV_max"] = static_cast<double>(dv.second);
		}
	}
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
		cout << result_field.join(",").toLocal8Bit().data() << std::endl;
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
		cout << std::setprecision(8);
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
						  const QString& param_header,
						  const QStringList& result_field,
						  const QStringList& axis_label)
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

	std::string	axistype = "loglogaxis";
	std::string	height = "6in";
	std::string	width = "6in";
	cout << "\\begin{tikzpicture}" << std::endl;
	cout << "\\begin{" << axistype << "}[" << std::endl;
	cout << "    height=" << height << "," << std::endl;
	cout << "    width=" << width << "," << std::endl;
	cout << "    xlabel=" << axis_label[0].toLocal8Bit().data() << "," << std::endl;
	cout << "    ylabel=" << axis_label[1].toLocal8Bit().data() << std::endl;
	cout << "]" << std::endl;

	for(size_t i = 0; i < result.size(); ++i)
	{
		cout << std::setw(10);
		cout << std::setprecision(8);
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
	cout << "\\end{" << axistype << "}" << std::endl;
	cout << "\\end{tikzpicture}" << std::endl;
}

void print_table(const std::vector<QVariantMap>& params,
				 const std::vector<QVariant>& variable,
				 const std::vector<std::vector<QVariantMap>>& result,
				 const QString& param_header,
				 const QStringList& result_field,
				 const QStringList& _axis_labels, const QString& format)
{
	QStringList	axis_labels(_axis_labels);
	if(axis_labels.isEmpty())
	{
		axis_labels = result_field;
	}

	if(format == "txt")
	{
		print_table_txt(params, variable, result, param_header, result_field);
	}
	else if(format == "pgfplots")
	{
		print_table_pgfplots(params, variable, result, param_header, result_field, axis_labels);
	}
	else
	{
		qDebug() << "Unknown table format" << format;
	}
}

int run_bench(const std::vector<QVariantMap>& params,
			  const std::vector<QVariant>& variable,
			  std::vector<std::vector<QVariantMap>>& result,
			  const QString& variable_field,
			  const QString& check_list,
			  nbcoord_t max_time)
{
	for(size_t i = 0; i < params.size(); ++i)
	{
		for(size_t j = 0; j < variable.size(); ++j)
		{
			QVariantMap	p(params[i]);
			p[variable_field] = variable[j];
			if(variable_field == "max_time")
			{
				max_time = variable[j].toDouble();
			}
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

#endif //NBODY_BENCH_H
