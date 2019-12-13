include( ../../pri/nbody.pri )


TEMPLATE	= app
TARGET		= nbody-solar
MOC_DIR = ./.tmp/moc
DESTDIR = ./

CONFIG		+= qt
QT += opengl

INCLUDEPATH += ../../nbody
INCLUDEPATH += ../bench
LIBS += -L../../lib
LIBS += -lnbody

SOURCES	+= main.cpp

