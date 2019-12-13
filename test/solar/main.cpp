#include <QCoreApplication>
#include <QDebug>
#include <omp.h>

#include "bench.h"
#include "nbody_data_stream.h"
#include "nbody_arg_parser.h"

void compute_end_state(const QString& initial_state,
					   const QString& end_state)
{
	QVariantMap param(std::map<QString, QVariant>(
	{
		{"name", "adams5"},
		{"engine", "simple"},
		{"solver", "adams"},
		{"max_step", 1.0 / 1024.0},
		{"rank", 5},
		{"starter_solver", "rkdp"},
		{"initial_state", initial_state},
		{"end_state", end_state},
	}));

	run(param, "PLVE", 365000);
}

void bench_solver(const QString& format,
				  const QString& initial_state,
				  const QString& expected_state)
{
	QString	engine("simple");
	QVariantMap param01(std::map<QString, QVariant>(
	{
		{"name", "adams5"},
		{"engine", engine},
		{"solver", "adams"},
		{"rank", 5},
		{"starter_solver", "rkdp"},
		{"initial_state", initial_state},
		{"expected_state", expected_state}
	}));
	QVariantMap param02(std::map<QString, QVariant>(
	{
		{"name", "rk4"},
		{"engine", engine},
		{"solver", "rk4"},
		{"initial_state", initial_state},
		{"expected_state", expected_state}
	}));
	QVariantMap param03(std::map<QString, QVariant>(
	{
		{"name", "rkdp"},
		{"engine", engine},
		{"solver", "rkdp"},
		{"initial_state", initial_state},
		{"expected_state", expected_state},
		{"min_step", "-1"}
	}));
	QVariantMap param04(std::map<QString, QVariant>(
	{
		{"name", "rkdverk"},
		{"engine", engine},
		{"solver", "rkdverk"},
		{"initial_state", initial_state},
		{"expected_state", expected_state},
		{"min_step", "-1"}
	}));

	std::vector<QVariantMap>				params = {param01, param02, param03, param04};
	std::vector<QVariant>					steps = {1.0, 1.0 / 4.0, 1.0 / 16.0, 1.0 / 64.0, 1.0 / 256.0};
	QString									variable_field = "max_step";
	std::vector<std::vector<QVariantMap>>	result(params.size(), std::vector<QVariantMap>(steps.size()));

	run_bench(params, steps, result, variable_field, "PLVE", 3000_f * 365.4_f);
	print_table(params, steps, result, "name", QStringList() << "CC" << "dE",
				QStringList() << "$f_n$ compute count" << "$dE/E_0$", format);
	print_table(params, steps, result, "name", QStringList() << "CC" << "dL",
				QStringList() << "$f_n$ compute count" << "$dL/L_0$", format);
	print_table(params, steps, result, "name", QStringList() << "CC" << "dP",
				QStringList() << "$f_n$ compute count" << "$dP/P_0$", format);
	print_table(params, steps, result, "name", QStringList() << "CC" << "dR",
				QStringList() << "$f_n$ compute count" << "$dR$", format);
	print_table(params, steps, result, "name", QStringList() << "CC" << "dV",
				QStringList() << "$f_n$ compute count" << "$dV$", format);
}


int main(int argc, char* argv[])
{
	QCoreApplication	a(argc, argv);
	QVariantMap			param(nbody_parse_arguments(argc, argv));
	const QString		bench(param.value("bench", "solver").toString());
	const QString		format(param.value("format", "txt").toString());

	if(bench == "compute_end_state")
	{
		QString	initial_state(param.value("initial_state", QString()).toString());
		QString	end_state(param.value("end_state", QString()).toString());

		if(initial_state.isEmpty())
		{
			qDebug() << "--initial_state must be set";
			return 1;
		}
		if(end_state.isEmpty())
		{
			qDebug() << "--end_state must be set";
			return 1;
		}

		compute_end_state(initial_state, end_state);
	}
	else if(bench == "solver")
	{
		QString	initial_state(param.value("initial_state", QString()).toString());
		QString	expected_state(param.value("expected_state", QString()).toString());

		if(initial_state.isEmpty())
		{
			qDebug() << "--initial_state must be set";
			return 1;
		}
		if(expected_state.isEmpty())
		{
			qDebug() << "--expected_state must be set";
			return 1;
		}
		bench_solver(format, initial_state, expected_state);
	}

	return 0;
}
