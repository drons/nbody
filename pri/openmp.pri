contains(DEFINES, HAVE_OPENMP){
	!clang{
		QMAKE_CXXFLAGS += -fopenmp
		QMAKE_CFLAGS += -fopenmp
		LIBS += -lgomp
	}
	clang{
		QMAKE_CXXFLAGS += -fopenmp
		QMAKE_CFLAGS += -fopenmp
		LIBS += -liomp5
	}
}
