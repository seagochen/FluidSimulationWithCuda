/**
* <Author>      Orlando Chen
* <First>       Jan 25, 2014
* <Last>		Jan 25, 2014
* <File>        Parameters.cpp
*/

#define DELTATIME           0.5f
#define STEPSIZE            0.001f
#define DIFFUSION           0.1f
#define VISOCITY            0.0f 
#define SOURCE              30

#define GRIDS_X             64
#define NODES_X             3
#define VOLUME_X            GRIDS_X*NODES_X
#define THREADS_X           512
#define TILE_X              16
#define WINDOWS_X           600
#define CANVAS_X            600
#define CUBESIZE_X          GRIDS_X*GRIDS_X*GRIDS_X
#define TPBUFFER_X          1024

#define BD_SOURCE          -1
#define BD_BLANK            0
#define BD_OBSTACLE         100

#define USING_GRID_DENS     0
#define USING_GRID_VELU     1
#define USING_GRID_VELV     2
#define USING_GRID_VELW     3

#define JACOBI_DENSITY      0
#define JACOBI_VELOCITY     1