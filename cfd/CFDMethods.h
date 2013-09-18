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
	
	/* 2D 紋理數據定義 */
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
	

	// 由二維速度場計算出經過時間timestep後，由速度平流影響到的新格點位置
	// 並使用Linear filtering對新坐標賦值
	Vector2d *Advect(double timestep, FLUIDSIM *in, Texture2D *velocity);

	// 計算散度
	double Divergence(FLUIDSIM *in, Texture2D *velocity);

	// Jacobi
	double Jacobi(FLUIDSIM *in, Texture2D *pressure, Texture2D *divergence);

	// 將advect divergence 及 Jacobi 等計算出的值用於計算結果 dt 後產生的新的速度場
	Vector2d *Project(FLUIDSIM *in, Texture2D *pressure, Texture2D *velocity);

};

#endif