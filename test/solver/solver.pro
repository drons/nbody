include( ../../pri/nbody.pri )

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
include( ../../pri/testdata.pri )
