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

#define VOLUME_X               128
#define VOLUME_Y               128
#define VOLUME_Z               128

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

#define MACRO_TRUE                0
#define MACRO_FALSE               1

#define BULLET_S             11

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


#define TEMPND_S             10

#define temp_dens            m_vectGlobalBufs[ 0 ]
#define temp_velu            m_vectGlobalBufs[ 1 ]
#define temp_velv            m_vectGlobalBufs[ 2 ]
#define temp_velw            m_vectGlobalBufs[ 4 ]
#define temp_obst            m_vectGlobalBufs[ 5 ]

#endif