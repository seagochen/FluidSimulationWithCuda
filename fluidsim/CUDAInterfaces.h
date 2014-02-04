/**
* <Author>        Orlando Chen
* <Email>         seagochen@gmail.com
* <First Time>    Feb 02, 2014
* <Last Time>     Feb 02, 2014
* <File Name>     CUDAInterfaces.h
*/

#ifndef __cuda_interfaces_h_
#define __cuda_interfaces_h_

#include "DataStructures.h"

using namespace sge;

/* ��CopyBuffer��C���Է�װ������������ݿ�������ʱ������ */
extern void hostCopyBuffer( SGTEMPBUFFERS *buffs, const SGCUDANODES *nodes, const SGFIELDTYPE type );

/* ��CopyBuffer��C���Է�װ������ʱ���ݿ����������� */
extern void hostCopyBuffer( SGCUDANODES *nodes, const SGTEMPBUFFERS *buffs, const SGFIELDTYPE type );

/* ��SwapBuffer��C���Է�װ����������GPU buffer�����ݣ���Ҫע��������������ݵĳ���Ӧ����һ���ģ���64^3 */
extern void hostSwapBuffer( SGTEMPBUFFERS *buf1, SGTEMPBUFFERS *buf2 );

/* ��ZeroBuffer��C���Է�װ����GPU buffer������������ */
extern void hostZeroBuffer( SGTEMPBUFFERS *buf );

/* �ɼ��������ݣ���ת��Ϊvolumetric data */
extern void hostPickData ( unsigned char *data, const SGTEMPBUFFERS *bufs,
	int const offi, int const offj, int const offk );

/* ����������м������� */
extern void hostAdvection( double *stores, SGTEMPBUFFERS *buff, SGCUDANODES *nodes );

/* ����ܶȳ� */
extern void DensitySolver
	( SGTEMPBUFFERS *dens, SGTEMPBUFFERS *dens0, SGCUDANODES *nodes, double *stores );

/* ����ٶȳ� */
extern void hostVelocitySolver
	( SGTEMPBUFFERS *u, SGTEMPBUFFERS *v, SGTEMPBUFFERS *w, SGTEMPBUFFERS *div, SGTEMPBUFFERS *p,
	SGTEMPBUFFERS *u0, SGTEMPBUFFERS *v0, SGTEMPBUFFERS *w0,
	SGCUDANODES *nodes, double *stores );

#endif