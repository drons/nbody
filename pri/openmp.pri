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
