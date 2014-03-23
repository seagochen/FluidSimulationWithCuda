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

#define and    &&
#define and_eq &=
#define bitand &
#define bitor  |
#define compl  ~
#define not    !
#define not_eq !=
#define or     ||
#define or_eq  |=
#define xor    ^
#define xor_eq ^=
#define eqt    ==
#define elif  else if

#endif