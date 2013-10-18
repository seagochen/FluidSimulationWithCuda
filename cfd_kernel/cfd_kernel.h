#ifndef __cfd_kernel_h_
#define __cfd_kernel_h_

/// DLL ///
#ifdef _In_Dll_File
#define _DLL __declspec(dllexport)    // Headers included in dll source
#else
#define _DLL __declspec(dllimport)    // Headers included from external  
#endif  

_DLL void print_cuda();

#endif