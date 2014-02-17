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
extern void hostCopyBuffer( SGSIMPLENODES *buffs, const SGCUDANODES *nodes, const SGFIELDTYPE type );

/* ��CopyBuffer��C���Է�װ������ʱ���ݿ����������� */
extern void hostCopyBuffer( SGCUDANODES *nodes, const SGSIMPLENODES *buffs, const SGFIELDTYPE type );

/* ��SwapBuffer��C���Է�װ����������GPU buffer�����ݣ���Ҫע��������������ݵĳ���Ӧ����һ���ģ���64^3 */
extern void hostSwapBuffer( SGSIMPLENODES *buf1, SGSIMPLENODES *buf2 );

/* ��ZeroBuffer��C���Է�װ����GPU buffer������������ */
extern void hostZeroBuffer( SGSIMPLENODES *buf );
extern void hostZeroBuffer( SGSTDGRID *buf );

/* �ɼ��������ݣ���ת��Ϊvolumetric data */
extern void hostPickData ( unsigned char *data, const SGSIMPLENODES *bufs,
	int const offi, int const offj, int const offk );

/* ����������м������� */
extern void hostAdvection( double *stores, SGSIMPLENODES *buff, SGCUDANODES *nodes );

/* ����ܶȳ� */
extern void DensitySolver
	( SGSIMPLENODES *dens, SGSIMPLENODES *dens0, SGCUDANODES *nodes, double *stores );

/* ����ٶȳ� */
extern void hostVelocitySolver
	( SGSIMPLENODES *u, SGSIMPLENODES *v, SGSIMPLENODES *w, SGSIMPLENODES *div, SGSIMPLENODES *p,
	SGSIMPLENODES *u0, SGSIMPLENODES *v0, SGSIMPLENODES *w0,
	SGCUDANODES *nodes, double *stores );

#endif