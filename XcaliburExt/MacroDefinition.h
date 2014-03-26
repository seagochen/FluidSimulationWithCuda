/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 23, 2014
* <Last Time>     Mar 25, 2014
* <File Name>     MacroDefiniton.h
*/

#ifndef __macro_definition_h_
#define __macro_definition_h_

/* parameters for fluid simulation */
#define DELTATIME              0.5f
#define DIFFUSION              0.1f
#define VISOCITY               0.0f
#define DENSITY               12.5f
#define VELOCITY              15.7f

#define GRIDS_X                 64
#define GRIDS_Y                 64
#define GRIDS_Z                 64

#define BULLET_X                66
#define BULLET_Y                66
#define BULLET_Z                66

/* dimension of simulation nodes */
#define NODES_X                  2
#define NODES_Y                  2
#define NODES_Z                  2

/* parameters for volume rendering */
#define STEPSIZE             0.001f

#define VOLUME_X                64
#define VOLUME_Y                64
#define VOLUME_Z                64

/* CUDA device's configuration info */
#define THREADS_S             1024

#define TILE_X                  32
#define TILE_Y                  32

/* screen resolution */
#define WINDOWS_X              480
#define WINDOWS_Y              480

#define CANVAS_X               480
#define CANVAS_Y               480

/* etc */
#define TPBUFFER_S            1024

#define APP_TITLE   "ExcaliburExt OTL 1.00.00. FPS: %d "

/* macro definition of grid types */
#define MACRO_DENSITY            0
#define MACRO_VELOCITY_U         1
#define MACRO_VELOCITY_V         2
#define MACRO_VELOCITY_W         3
#define MACRO_SIMPLE             4

/* macro definition of boundary condition */
#define MACRO_BOUNDARY_BLANK      0
#define MACRO_BOUNDARY_OBSTACLE 100
#define MACRO_BOUNDARY_SOURCE  -100

/* macro definition of node's position */
#define MACRO_CENTER              0
#define MACRO_LEFT                1
#define MACRO_RIGHT               2
#define MACRO_UP                  3
#define MACRO_DOWN                4
#define MACRO_FRONT               5
#define MACRO_BACK                6


#define BULLET_S             40

#define dev_den              m_vectCompBufs[ 0 ]
#define dev_den0             m_vectCompBufs[ 1 ]
#define dev_u                m_vectCompBufs[ 2 ]
#define dev_u0               m_vectCompBufs[ 3 ]
#define dev_v                m_vectCompBufs[ 4 ]
#define dev_v0               m_vectCompBufs[ 5 ]
#define dev_w                m_vectCompBufs[ 6 ]
#define dev_w0               m_vectCompBufs[ 7 ]
#define dev_div              m_vectCompBufs[ 8 ]
#define dev_p                m_vectCompBufs[ 9 ]
#define dev_obs              m_vectCompBufs[ 10 ]

#define dens_C               m_vectCompBufs[ 0 ]
#define dens_L               m_vectCompBufs[ 11 ]
#define dens_R               m_vectCompBufs[ 12 ]
#define dens_U               m_vectCompBufs[ 13 ]
#define dens_D               m_vectCompBufs[ 14 ]
#define dens_F               m_vectCompBufs[ 15 ]
#define dens_B               m_vectCompBufs[ 16 ]

#define velu_C               m_vectCompBufs[ 2 ]
#define velu_L               m_vectCompBufs[ 17 ] 
#define velu_R               m_vectCompBufs[ 18 ]
#define velu_U               m_vectCompBufs[ 19 ]
#define velu_D               m_vectCompBufs[ 20 ]
#define velu_F               m_vectCompBufs[ 21 ]
#define velu_B               m_vectCompBufs[ 22 ]

#define velv_C               m_vectCompBufs[ 4 ]
#define velv_L               m_vectCompBufs[ 23 ]
#define velv_R               m_vectCompBufs[ 24 ]
#define velv_U               m_vectCompBufs[ 25 ]
#define velv_D               m_vectCompBufs[ 26 ]
#define velv_F               m_vectCompBufs[ 27 ]
#define velv_B               m_vectCompBufs[ 28 ]

#define velw_C               m_vectCompBufs[ 6 ]
#define velw_L               m_vectCompBufs[ 29 ]
#define velw_R               m_vectCompBufs[ 30 ]
#define velw_U               m_vectCompBufs[ 31 ]
#define velw_D               m_vectCompBufs[ 32 ]
#define velw_F               m_vectCompBufs[ 33 ]
#define velw_B               m_vectCompBufs[ 34 ]


#define TEMPND_S             10

#define temp_dens            m_vectGlobalBufs[ 0 ]
#define temp_velu            m_vectGlobalBufs[ 1 ]
#define temp_velv            m_vectGlobalBufs[ 2 ]
#define temp_velw            m_vectGlobalBufs[ 4 ]
#define temp_obst            m_vectGlobalBufs[ 5 ]

#endif