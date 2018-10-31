#include <QApplication>
#include <QDebug>
#include <omp.h>

#include "wgt_nbody_player.h"

#include "nbody_solvers.h"
#include "nbody_engines.h"

int main(int argc, char* argv[])
{
	QApplication		app(argc, argv);

	wgt_nbody_player*	nbv = new wgt_nbody_player();

	nbv->show();

	return app.exec();
}
