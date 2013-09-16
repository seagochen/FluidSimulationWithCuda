#include "CFDMethods.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <SGE\SGUtils.h>

#include <sstream>
#include <iostream>

using namespace sge;
using std::stringstream;
using std::endl;

Texture2D velocity;
Texture2D pressure;
Texture2D divergence;

FileManager file;
stringstream buffer;

void InitVelocity(), InitPressure(), InitDivergence();

FLUIDSIM index[SIMAREA_WIDTH][SIMAREA_HEIGHT];

void InitIndex();

int main()
{
	srand(time(NULL));

	// Init texture 2Ds
	InitVelocity();
	InitPressure();
	InitDivergence();
	InitIndex();

	system("pause");
};

void InitVelocity()
{
	buffer.str("");
	buffer<<"------------------------------------------Init------------------------------------------"<<endl;

	for (int u=0; u<SIMAREA_WIDTH; u++)
	{
		for (int v=0; v<SIMAREA_HEIGHT; v++)
		{
			velocity.uv[u][v][0] = (double)rand() / (double)RAND_MAX;
			velocity.uv[u][v][1] = (double)rand() / (double)RAND_MAX;
			buffer<<velocity.uv[u][v][0]<<"  "<<velocity.uv[u][v][1]<<"  |  ";
		}
		buffer<<endl;
	}
	file.SetDataToFile(buffer.str(), "velocity.txt", SGFILEOPENTYPE::SG_FILE_OPEN_DEFAULT);
};

void InitPressure()
{
	buffer.str("");
	buffer<<"------------------------------------------Init------------------------------------------"<<endl;

	for (int u=0; u<SIMAREA_WIDTH; u++)
	{
		for (int v=0; v<SIMAREA_HEIGHT; v++)
		{
			pressure.s[u][v] = (double)rand() / (double)RAND_MAX;
			buffer<<pressure.s[u][v]<<"    ";
		}
		buffer<<endl;
	}
	file.SetDataToFile(buffer.str(), "pressure.txt", SGFILEOPENTYPE::SG_FILE_OPEN_DEFAULT);
};

void InitDivergence()
{
	buffer.str("");
	buffer<<"------------------------------------------Init------------------------------------------"<<endl;

	for (int u=0; u<SIMAREA_WIDTH; u++)
	{
		for (int v=0; v<SIMAREA_HEIGHT; v++)
		{
			divergence.s[u][v] = (double)rand() / (double)RAND_MAX;
			buffer<<divergence.s[u][v]<<"    ";
		}
		buffer<<endl;
	}
	file.SetDataToFile(buffer.str(), "divergence.txt", SGFILEOPENTYPE::SG_FILE_OPEN_DEFAULT);
};

void InitIndex()
{
	buffer.str("");
	for (int u=0; u<SIMAREA_WIDTH; u++)
	{
		for (int v=0; v<SIMAREA_HEIGHT; v++)
		{
			index[u][v].CellIndex(u, v);
			index[u][v].CenterCell(u, v);

			// Fill the neighboring cells index
			int ut = u; int vt =v;
			if (ut - 1 < 0) ut = 0;
			index[u][v].LeftCell(ut, vt);
			if (ut + 1 >= SIMAREA_WIDTH) ut = SIMAREA_WIDTH - 1;
			index[u][v].RightCell(ut, vt);
			if (vt - 1 < 0) vt = 0;
			index[u][v].DownCell(ut, vt);
			if (vt + 1 >= SIMAREA_HEIGHT) vt = SIMAREA_HEIGHT - 1;
			index[u][v].UpCell(ut, vt);

			buffer<<"# "<<u<<v<<
				"  L: "<<index[u][v].LeftCell[0]<<index[u][v].LeftCell[1]<<
				"  R: "<<index[u][v].RightCell[0]<<index[u][v].RightCell[1]<<
				"  U: "<<index[u][v].UpCell[0]<<index[u][v].UpCell[1]<<
				"  B: "<<index[u][v].DownCell[0]<<index[u][v].DownCell[1]<<endl;
		}
		buffer<<"-----------------------------------------------------------------"<<endl<<endl;
	}

	file.SetDataToFile(buffer.str(), "index.txt", SGFILEOPENTYPE::SG_FILE_OPEN_DEFAULT);
};