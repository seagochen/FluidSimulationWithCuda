/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Jan 23, 2014
* <Last Time>     Feb 17, 2014
* <File Name>     MacroDefiniton.h
*/

#ifndef __macro_definition_h_
#define __macro_definition_h_

#define DELTATIME              0.5f   // ����0.5sΪһ��������delta time
#define STEPSIZE               0.001f // ����0.001Ϊһ���������
#define DIFFUSION              0.1f   // diffusion�Ķ���ֵΪ0.1
#define VISOCITY               0.0f   // visocity�Ķ���ֵΪ0.1
#define SOURCE_DENSITY        100     // Ϊ������������ӵ�density��Ũ��
#define SOURCE_VELOCITY       100     // Ϊ������������ӵ�velocity����

#define GRIDS_X                64     // ���������ڵ�ά������ӵ�е�����
#define NODES_X                 2     // ����ڵ��ڵ�ά������ӵ�е�����
#define VOLUME_X    GRIDS_X*NODES_X   // ��ά�������ڵ�ά���ϵĳ���
#define THREADS_X             1024    // ����CUDA���߳�����
#define TILE_X                 16     // ��16x16��GPU-threads������Ϊһ��block
#define WINDOWS_X              600    // Windows application's size
#define CANVAS_X               600    // canvas's size
#define TPBUFFER_X             1024   // Ϊ�˿���ֲ�Զ���������ʱ���ݻ��棬�������shared memories

#endif