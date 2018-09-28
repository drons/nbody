!clang {
	NBODY_FLAGS += -ftree-vectorizer-verbose=1
	NBODY_FLAGS += -mavx -ffast-math -finline-functions -funswitch-loops -fpredictive-commoning -fgcse-after-reload -ftree-vectorize -fipa-cp-clone
}
QMAKE_CFLAGS += $$NBODY_FLAGS
QMAKE_CXXFLAGS += $$NBODY_FLAGS
