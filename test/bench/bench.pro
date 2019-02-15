include( ../../pri/nbody.pri )


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

