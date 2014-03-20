/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 23, 2014
* <Last Time>     Mar 19, 2014
* <File Name>     MacroDefiniton.h
*/

#ifndef __macro_definition_h_
#define __macro_definition_h_

#include <device_launch_parameters.h>
#include <string>

typedef double const  cdouble;
typedef int const     cint;
typedef unsigned char uchar;
typedef std::string*  sstr;

/* parameters for fluid simulation */
#define DELTATIME              0.5f
#define DIFFUSION              0.1f
#define VISOCITY               0.0f
#define DENSITY               12.5f
#define VELOCITY              15.7f

#define GRIDS_X                  64
#define GRIDS_Y                  64
#define GRIDS_Z                  64

#define BULLET_S                 40

#define BULLET_X                 66
#define BULLET_Y                 66
#define BULLET_Z                 66

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

/* True and False */
#define MACRO_FALSE               0
#define MACRO_TRUE                1

/* switch */
#define TESTING_MODE_SWITCH       0 /* switch: close(0) open(1) */
#define TESTING_MODE              0 /* velocity: default-up(0) down(1) left(2) right(3) front(4) back(5) */

#define gst_header                0  /* (ghost, halo) the header cell of grid */
#define sim_header                1  /* (actually) the second cell of grid */
#define gst_tailer               63  /* (ghost, halo) the last cell of grid */
#define sim_tailer               62  /* (actually) the second last cell of grid */

#endif