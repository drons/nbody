include( ../../pri/opencl.pri )
include( ../../pri/vectorize.pri )

TEMPLATE	= app
TARGET		= engine
MOC_DIR = ./.tmp/moc

CONFIG		+= qt
QT += opengl
LIBS += -lGLU -lgomp

INCLUDEPATH += ../../nbody
LIBS += -L../../nbody
LIBS += -lnbody

SOURCES	+= main.cpp

#HEADERS	+=



