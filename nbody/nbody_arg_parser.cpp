#include "nbody_arg_parser.h"
#include <QDebug>
#include <QStringList>

QVariantMap nbody_parse_arguments(int argc, char** argv)
{
	QVariantMap		param;
	const QString	arg_prefix("--");

	for(int arg_n = 1; arg_n < argc; ++arg_n)
	{
		QString		arg(argv[arg_n]);
		if(!arg.startsWith(arg_prefix))
		{
			continue;
		}

		QStringList	p(arg.mid(arg_prefix.length()).split("="));
		if(p.size() != 2)
		{
			qDebug() << "Invalid argument format" << arg;
			continue;
		}
		param[p[0]] = p[1];
	}

	return param;
}
