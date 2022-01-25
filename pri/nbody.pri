CONFIG += depend_includepath
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
