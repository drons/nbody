#ifdef NBODY_EXPORT_DLL
#if defined( WIN32 )
#define NBODY_DLL	__declspec(dllexport)
#else
#define NBODY_DLL
#endif
#else// #ifdef NBODY_EXPORT_DLL
#if defined( WIN32 )
#define NBODY_DLL	__declspec(dllimport)
#else
#define NBODY_DLL
#endif
#endif// #ifdef NBODY_EXPORT_DLL
