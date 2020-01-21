#include <QCoreApplication>
#include <QDebug>
#include <omp.h>

#include "bench.h"
#include "nbody_data_stream.h"
#include "nbody_arg_parser.h"
#include "nbody_step_visitor.h"

int compute_end_state(const QVariantMap& param)
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
	run(param, "PLVE", 400 * 365.25);
	return 0;
}

void bench_solver(const QString& format,
				  const QString& initial_state,
				  const QString& expected_state)
{
	QString	engine("openmp");
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
		{"name", "adams5-corr"},
		{"engine", engine},
		{"solver", "adams"},
		{"rank", 5},
		{"correction", true},
		{"starter_solver", "rkdp"},
		{"initial_state", initial_state},
		{"expected_state", expected_state}
	}));
	QVariantMap param03(std::map<QString, QVariant>(
	{
		{"name", "rkdp-fixed-step"},
		{"engine", engine},
		{"solver", "rkdp"},
		{"initial_state", initial_state},
		{"expected_state", expected_state},
		{"min_step", "-1"}
	}));
	QVariantMap param04(std::map<QString, QVariant>(
	{
		{"name", "rkdp-fixed-step-corr"},
		{"engine", engine},
		{"solver", "rkdp"},
		{"correction", true},
		{"initial_state", initial_state},
		{"expected_state", expected_state},
		{"min_step", "-1"}
	}));
	QVariantMap param05a(std::map<QString, QVariant>(
	{
		{"name", "midpoint"},
		{"engine", engine},
		{"solver", "midpoint"},
		{"initial_state", initial_state},
		{"expected_state", expected_state},
		{"min_step", "-1"}
	}));
	QVariantMap param05b(std::map<QString, QVariant>(
	{
		{"name", "midpoint-st"},
		{"engine", engine},
		{"solver", "midpoint-st"},
		{"initial_state", initial_state},
		{"expected_state", expected_state},
		{"min_step", "-1"}
	}));
	QVariantMap param06(std::map<QString, QVariant>(
	{
		{"name", "bs4-fixed-step"},
		{"engine", engine},
		{"solver", "bs"},
		{"max_level", 4},
		{"error_threshold", 1e-14},
		{"initial_state", initial_state},
		{"expected_state", expected_state},
		{"min_step", "-1"}
	}));
	QVariantMap param07(std::map<QString, QVariant>(
	{
		{"name", "bs8-fixed-step"},
		{"engine", engine},
		{"solver", "bs"},
		{"max_level", 8},
		{"error_threshold", 1e-14},
		{"initial_state", initial_state},
		{"expected_state", expected_state},
		{"min_step", "-1"}
	}));
	QVariantMap param08(std::map<QString, QVariant>(
	{
		{"name", "bs16-fixed-step"},
		{"engine", engine},
		{"solver", "bs"},
		{"max_level", 16},
		{"error_threshold", 1e-14},
		{"initial_state", initial_state},
		{"expected_state", expected_state},
		{"min_step", "-1"}
	}));
	QVariantMap param09(std::map<QString, QVariant>(
	{
		{"name", "rkdverk-fixed-step"},
		{"engine", engine},
		{"solver", "rkdverk"},
		{"correction", false},
		{"initial_state", initial_state},
		{"expected_state", expected_state},
		{"min_step", "-1"}
	}));
	QVariantMap param10(std::map<QString, QVariant>(
	{
		{"name", "rkdverk-fixed-step-corr"},
		{"engine", engine},
		{"solver", "rkdverk"},
		{"correction", true},
		{"initial_state", initial_state},
		{"expected_state", expected_state},
		{"min_step", "-1"}
	}));
	QVariantMap param11(std::map<QString, QVariant>(
	{
		{"name", "rkf-fixed-step"},
		{"engine", engine},
		{"solver", "rkf"},
		{"correction", false},
		{"initial_state", initial_state},
		{"expected_state", expected_state},
		{"min_step", "-1"}
	}));
	QVariantMap param12(std::map<QString, QVariant>(
	{
		{"name", "rkf-fixed-step-corr"},
		{"engine", engine},
		{"solver", "rkf"},
		{"correction", true},
		{"initial_state", initial_state},
		{"expected_state", expected_state},
		{"min_step", "-1"}
	}));

	std::vector<QVariantMap>				params = {param01, param02, param03, param04, param05a, param05b, param06, param07, param08, param09, param10, param11, param12};
	std::vector<QVariant>					steps = {16.0, 4.0, 1.0, 1.0 / 4.0, 1.0 / 16.0, 1.0 / 64.0, 1.0 / 256.0};
	QString									variable_field = "max_step";
	std::vector<std::vector<QVariantMap>>	result(params.size(), std::vector<QVariantMap>(steps.size()));

	run_bench(params, steps, result, variable_field, "PLVE", 3000_f * 365.25_f);
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

class nbody_step_visitor_comparator : public nbody_step_visitor
{
	nbody_data								m_expected_data;
	std::vector<std::vector<QVariantMap>>	m_result;
	std::vector<QVariant>					m_variable;
	std::vector<QVariantMap>				m_params;
	nbcoord_t								m_min_dr;
	nbcoord_t								m_min_dr_time;
public:
	nbody_step_visitor_comparator() :
		m_min_dr(1e30_f),
		m_min_dr_time(0_f)
	{
	}
	~nbody_step_visitor_comparator()
	{
		std::cout << "%" << std::setw(10) << std::setprecision(8)
				  << "Min dR=" << static_cast<double>(m_min_dr)
				  << " T=" << static_cast<double>(m_min_dr_time);
	}
	bool load_init(const QString& expected_state)
	{
		bool res = m_expected_data.load_initial(expected_state, "G1");
		size_t	count = m_expected_data.get_count();
		m_result.resize(count + 1);
		m_params.resize(count + 1);
		for(size_t i = 0; i != count; ++i)
		{
			m_params[i]["name"] = QString::number(i);
		}
		m_params[count]["name"] = "mean";
		return res;
	}
	void push(size_t n, double t, double dr)
	{
		QVariantMap	r;
		r["T"] = t;
		r["dR"] = dr;
		m_result[n].push_back(r);
	}
	void visit(const nbody_data* data) override
	{
		double	t = static_cast<double>(data->get_time()) / 365;
		size_t	count = data->get_count();
		auto* vert = data->get_vertites();
		auto* expected_vert = m_expected_data.get_vertites();
		const auto dr = compare_data(vert, expected_vert, count);
		if(t > 3002)
		{
			for(size_t i = 0; i != count; ++i)
			{
				push(i, t, vert[i].distance(expected_vert[i]));
			}
			push(count, t, dr.first);
			m_variable.push_back(t);
		}
		if(m_min_dr > dr.first)
		{
			m_min_dr = dr.first;
			m_min_dr_time = t;
		}
	}
	const std::vector<std::vector<QVariantMap>> result() const
	{
		return m_result;
	}
	const std::vector<QVariant> variable() const
	{
		return m_variable;
	}
	const std::vector<QVariantMap> params() const
	{
		return m_params;
	}
};

void plot_period(const QString& format,
				 const QString& initial_state,
				 const QString& expected_state,
				 const QVariantMap& args)
{
	QString	engine("openmp");
	QVariantMap param01(std::map<QString, QVariant>(
	{
		{"verbose", 0},
		{"engine", engine},
		{"starter_solver", "rkdp"},
		{"min_step", -1},
		{"initial_state", initial_state}
	}));

	param01.unite(args);
	std::shared_ptr<nbody_step_visitor_comparator>	checker =
		std::make_shared<nbody_step_visitor_comparator>();
	checker->load_init(expected_state);

	run(param01, "E", 365 * 3004, checker);
	const auto	result(checker->result());
	const auto	variable(checker->variable());
	const auto	params(checker->params());
	QVariantMap	plot_param;
	plot_param["ymode"] = "log";
	plot_param["mark"] = "none";
	for(auto ii = param01.begin(); ii != param01.end(); ++ii)
	{
		std::cout << "%%" << ii.key().toLocal8Bit().data() << "="
				  << ii.value().toByteArray().data() << "," << std::endl;
	}
	print_table(params, variable, result, "name", QStringList() << "T" << "dR",
				QStringList() << "$T$" << "$dR$", format, plot_param);
}

class nbody_step_visitor_orbit_dump : public nbody_step_visitor
{
	std::vector<std::vector<QVariantMap>>	m_result;
	std::vector<QVariant>					m_variable;
	std::vector<QVariantMap>				m_params;
public:
	nbody_step_visitor_orbit_dump()
	{
	}
	~nbody_step_visitor_orbit_dump()
	{
	}
	void push(size_t n, double t, double dr)
	{
		QVariantMap	r;
		r["T"] = t;
		r["dR"] = dr;
		m_result[n].push_back(r);
	}
	void visit(const nbody_data* data) override
	{
		double	t = static_cast<double>(data->get_time()) / 365.0;
		size_t	count = data->get_count();
		if(m_result.empty())
		{
			m_result.resize(count);
			m_params.resize(count);
			for(size_t i = 0; i != count; ++i)
			{
				m_params[i]["name"] = QString::number(i);
			}
		}
		auto* vert = data->get_vertites();
		for(size_t i = 0; i != count; ++i)
		{
			push(i, t, vert[i].length());
		}
		m_variable.push_back(t);
	}
	const std::vector<std::vector<QVariantMap>> result() const
	{
		return m_result;
	}
	const std::vector<QVariant> variable() const
	{
		return m_variable;
	}
	const std::vector<QVariantMap> params() const
	{
		return m_params;
	}
};

void plot_start_period(const QString& format,
					   const QString& initial_state)
{
	QString	engine("simple");
	QVariantMap param01(std::map<QString, QVariant>(
	{
		{"verbose", 0},
		{"name", "rkdp"},
		{"engine", engine},
		{"solver", "rkdp"},
		{"starter_solver", "rkdp"},
		{"max_step", 1.0 / 4.0},
		{"min_step", -1},
		{"check_step", 10.0},
		{"initial_state", initial_state}
	}));

	std::shared_ptr<nbody_step_visitor_orbit_dump>	checker =
		std::make_shared<nbody_step_visitor_orbit_dump>();

	run(param01, "E", 365 * 100, checker);
	const auto	result(checker->result());
	const auto	variable(checker->variable());
	const auto	params(checker->params());
	QVariantMap	plot_param;
	plot_param["ymode"] = "log";
	plot_param["mark"] = "none";
	for(auto ii = param01.begin(); ii != param01.end(); ++ii)
	{
		std::cout << "%%" << ii.key().toLocal8Bit().data() << "="
				  << ii.value().toByteArray().data() << "," << std::endl;
	}
	print_table(params, variable, result, "name", QStringList() << "T" << "dR",
				QStringList() << "$T$" << "$R$", format, plot_param);
}

int main(int argc, char* argv[])
{
	QCoreApplication	a(argc, argv);
	QVariantMap			param(nbody_parse_arguments(argc, argv));
	const QString		bench(param.value("bench", "solver").toString());
	const QString		format(param.value("format", "txt").toString());

	if(bench == "compute_end_state")
	{
		return compute_end_state(param);
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
	else if(bench == "plot_period")
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
		plot_period(format, initial_state, expected_state, param);
	}
	else if(bench == "plot_start_period")
	{
		QString	initial_state(param.value("initial_state", QString()).toString());

		if(initial_state.isEmpty())
		{
			qDebug() << "--initial_state must be set";
			return 1;
		}
		plot_start_period(format, initial_state);
	}
	return 0;
}
