/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Mar 19, 2014
* <Last Time>     Mar 19, 2014
* <File Name>     SimulationParam.h
*/

#ifndef __sim_param_h__
#define __sim_param_h__

#include "FluidSimProc.h"
#include <vector>
#include <string>
#include <device_launch_parameters.h>

#define dev_den              m_vectGPUBuffers[ 0 ]
#define dev_den0             m_vectGPUBuffers[ 1 ]
#define dev_u                m_vectGPUBuffers[ 2 ]
#define dev_u0               m_vectGPUBuffers[ 3 ]
#define dev_v                m_vectGPUBuffers[ 4 ]
#define dev_v0               m_vectGPUBuffers[ 5 ]
#define dev_w                m_vectGPUBuffers[ 6 ]
#define dev_w0               m_vectGPUBuffers[ 7 ]
#define dev_div              m_vectGPUBuffers[ 8 ]
#define dev_p                m_vectGPUBuffers[ 9 ]
#define dev_obs              m_vectGPUBuffers[ 10 ]

#define dens_C               m_vectGPUBuffers[ 0 ]
#define dens_L               m_vectGPUBuffers[ 11 ]
#define dens_R               m_vectGPUBuffers[ 12 ]
#define dens_U               m_vectGPUBuffers[ 13 ]
#define dens_D               m_vectGPUBuffers[ 14 ]
#define dens_F               m_vectGPUBuffers[ 15 ]
#define dens_B               m_vectGPUBuffers[ 16 ]

#define velu_C               m_vectGPUBuffers[ 2 ]
#define velu_L               m_vectGPUBuffers[ 17 ] 
#define velu_R               m_vectGPUBuffers[ 18 ]
#define velu_U               m_vectGPUBuffers[ 19 ]
#define velu_D               m_vectGPUBuffers[ 20 ]
#define velu_F               m_vectGPUBuffers[ 21 ]
#define velu_B               m_vectGPUBuffers[ 22 ]

#define velv_C               m_vectGPUBuffers[ 4 ]
#define velv_L               m_vectGPUBuffers[ 23 ]
#define velv_R               m_vectGPUBuffers[ 24 ]
#define velv_U               m_vectGPUBuffers[ 25 ]
#define velv_D               m_vectGPUBuffers[ 26 ]
#define velv_F               m_vectGPUBuffers[ 27 ]
#define velv_B               m_vectGPUBuffers[ 28 ]

#define velw_C               m_vectGPUBuffers[ 6 ]
#define velw_L               m_vectGPUBuffers[ 29 ]
#define velw_R               m_vectGPUBuffers[ 30 ]
#define velw_U               m_vectGPUBuffers[ 31 ]
#define velw_D               m_vectGPUBuffers[ 32 ]
#define velw_F               m_vectGPUBuffers[ 33 ]
#define velw_B               m_vectGPUBuffers[ 34 ]

#endif