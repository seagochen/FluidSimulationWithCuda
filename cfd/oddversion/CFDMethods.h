#ifndef _SEAGOSOFT_CFDMETHODS_H_
#define _SEAGOSOFT_CFDMETHODS_H_

#define SIMAREA_WIDTH  256
#define SIMAREA_HEIGHT 256

#define SIMAREA_SWIDTH  10
#define SIMAREA_SHEIGHT 10

#include <Eigen\Dense>

#define samPointClamp  0
#define samLinear      1

namespace sge 
{
	using Eigen::Vector2d;
	
	struct FLUIDSIM
	{
		Vector2d CellIndex;
		Vector2d CenterCell;
		Vector2d LeftCell;
		Vector2d RightCell;
		Vector2d UpCell;
		Vector2d DownCell;
	};
	
	/* 2D �y�픵�����x */
	class Texture2D
	{
	public:
		// Vector fields
		double u[SIMAREA_WIDTH];
		double v[SIMAREA_HEIGHT];
		// Scalar fields
		double s[SIMAREA_WIDTH][SIMAREA_HEIGHT];

	public:
		// Return a vector
		Vector2d *Sample(int Sampling, Vector2d *CellIndex);
		// Return a scalar
		double SampleData(int Sampling, Vector2d *CellIndex);

	private:
		Vector2d *SamplePoint(int u, int v);
	};
	

	// �ɶ��S�ٶȈ�Ӌ������^�r�gtimestep�ᣬ���ٶ�ƽ��Ӱ푵����¸��cλ��
	// �Kʹ��Linear filtering���������xֵ
	Vector2d *Advect(double timestep, FLUIDSIM *in, Texture2D *velocity);

	// Ӌ��ɢ��
	double Divergence(FLUIDSIM *in, Texture2D *velocity);

	// Jacobi
	double Jacobi(FLUIDSIM *in, Texture2D *pressure, Texture2D *divergence);

	// ��advect divergence �� Jacobi ��Ӌ�����ֵ���Ӌ��Y�� dt ��a�����µ��ٶȈ�
	Vector2d *Project(FLUIDSIM *in, Texture2D *pressure, Texture2D *velocity);

};

#endif