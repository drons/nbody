include( ../../pri/nbody.pri )
include( ../../pri/opencl.pri )
include( ../../pri/vectorize.pri )

TEMPLATE	= app
TARGET		= nbody-bench
MOC_DIR = ./.tmp/moc
DESTDIR = ./

CONFIG		+= qt
QT += opengl

INCLUDEPATH += ../../nbody
LIBS += -L../../lib
LIBS += -lnbody

SOURCES	+= main.cpp

