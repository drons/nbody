CONFIG += depend_includepath

DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0

contains(CONFIG, USE_STATICLIB){
	!win32{
		QMAKE_LFLAGS += -fuse-ld=gold
	}
	DEFINES += HAVE_NBODY_STATICLIB
}

unix{
	include(gcc.pri)
}

include(openmp.pri)
include(vectorize.pri)
!contains(DEFINES,NB_COORD_PRECISION=4){
	include(opencl.pri)
	include(cuda.pri)
}
contains(DEFINES,NB_COORD_PRECISION=4){
	LIBS += -lquadmath
}
