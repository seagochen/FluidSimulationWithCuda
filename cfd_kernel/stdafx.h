// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
// Windows Header Files:

#include <windows.h>
#include <Winuser.h>
#include <stdexcept>
#include <atltypes.h>

// TODO: reference additional headers your program requires here
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cfd_kernel.h"