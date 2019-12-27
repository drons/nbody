unix{
	!clang {
		NBODY_FLAGS += -ftree-vectorizer-verbose=1
		NBODY_FLAGS += -O3 -mavx -ftree-vectorize
	}
	QMAKE_CFLAGS_RELEASE -= -O2
	QMAKE_CXXFLAGS_RELEASE -= -O2
	QMAKE_CFLAGS_RELEASE += $$NBODY_FLAGS
	QMAKE_CXXFLAGS_RELEASE += $$NBODY_FLAGS
}
