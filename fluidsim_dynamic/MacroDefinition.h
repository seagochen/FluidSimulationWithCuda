/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 23, 2014
* <Last Time>     Feb 17, 2014
* <File Name>     MacroDefiniton.h
*/

#ifndef __macro_definition_h_
#define __macro_definition_h_

#define DELTATIME              0.5f   // 定义0.5s为一个步长的delta time
#define STEPSIZE               0.001f // 定义0.001为一个步长深度
#define DIFFUSION              0.1f   // diffusion的定义值为0.1
#define VISOCITY               0.0f   // visocity的定义值为0.1
#define SOURCE_DENSITY        100     // 为计算网格中添加的density的浓度
#define SOURCE_VELOCITY       100     // 为计算网格中添加的velocity的量

#define GRIDS_X                64     // 计算网格在单维度上所拥有的数量
#define NODES_X                 2     // 计算节点在单维度上所拥有的数量
#define VOLUME_X    GRIDS_X*NODES_X   // 三维体数据在单维度上的长度
#define THREADS_X             1024    // 定义CUDA的线程数量
#define TILE_X                 16     // 将16x16的GPU-threads捆绑打包为一个block
#define WINDOWS_X              600    // Windows application's size
#define CANVAS_X               600    // canvas's size
#define TPBUFFER_X             1024   // 为了可移植性而创建的临时数据缓存，用于替代shared memories

#endif