include( ../../pri/nbody.pri )
include( ../../pri/opencl.pri )
include( ../../pri/cuda.pri )
include( ../../pri/vectorize.pri )

TEMPLATE	= app
TARGET		= engine
MOC_DIR = ./.tmp/moc
DESTDIR = ./

CONFIG		+= qt testcase
QT += opengl testlib
win32:LIBS += -lGLU32
unix:LIBS += -lGLU

INCLUDEPATH += ../../nbody
LIBS += -L../../lib
LIBS += -lnbody

SOURCES	+= test_nbody_engine.cpp


