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
	
	/* 定義二維速度場 */
	class Velocity2D
	{
	public:
		double velocity_u[SIMAREA_WIDTH];
		double velocity_v[SIMAREA_HEIGHT];

	public:
		Vector2d *Sample(int Sampling, Vector2d *CellIndex);

	private:
		Vector2d *SamplePoint(int u, int v);
	};
	

	// 由二維速度場計算出經過時間timestep後，由速度平流影響到的新格點位置
	// 並使用Linear filtering對新坐標賦值
	Vector2d *advect(double timestep, FLUIDSIM *in, Velocity2D *velocity);

};

#endif