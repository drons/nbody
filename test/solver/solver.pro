include( ../../pri/nbody.pri )
include( ../../pri/opencl.pri )
include( ../../pri/vectorize.pri )

TEMPLATE	= app
TARGET		= solver
MOC_DIR = ./.tmp/moc
DESTDIR = ./

CONFIG		+= qt testcase
QT += opengl testlib
win32:LIBS += -lGLU32
unix:LIBS += -lGLU

INCLUDEPATH += ../../nbody
LIBS += -L../../lib
LIBS += -lnbody

SOURCES += test_nbody_solver.cpp

TEST_DATA += \
    ../data/adams5.txt \
    ../data/euler.txt \
    ../data/initial_state.txt \
    ../data/rk4.txt \
    ../data/rkck.txt \
    ../data/rkdp.txt \
    ../data/rkdverk.txt \
    ../data/rkf.txt \
    ../data/rkgl.txt \
    ../data/rklc.txt \
    ../data/trapeze2.txt
{
	testdata.name = Copy test data
	testdata.input = TEST_DATA
	testdata.CONFIG += no_link target_predeps

	testdata.commands = $$QMAKE_COPY ${QMAKE_FILE_IN} ${QMAKE_FILE_OUT}
	testdata.output = $$DESTDIR/../data/${QMAKE_FILE_BASE}${QMAKE_FILE_EXT}

	QMAKE_EXTRA_COMPILERS += testdata
}
