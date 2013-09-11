#include "Headers.h"
#include "Visualization.h"
#include <time.h>

using namespace sge;

_volumeData            vbdata;
extern Visualization   *visual;

DWORD simulation (LPVOID lpdwThreadParam )
{
	vbdata.width = 256;
	vbdata.height = 256;
	vbdata.depth = 256;
	vbdata.textureID = 10;

	void update();

	vbdata.data = (unsigned char*) malloc (
		sizeof(unsigned char) * vbdata.width * vbdata.height * vbdata.depth * BYTES_PER_TEXEL);

	srand(time(NULL));

	while( TRUE )
	{
		visual->LoadVolumeData(&vbdata);
		update();
	}
};

void update()
{
	for (int i=0; i<vbdata.width; i++) {
		for (int j=0; j<vbdata.height; j++) {
			for (int k=0; k<vbdata.depth; k++) {
				vbdata.data[visual->TEXEL3(i, j, k)]  = 255;  // red
				vbdata.data[visual->TEXEL3(i, j, k) + 1]  = 0;  // green
				vbdata.data[visual->TEXEL3(i, j, k) + 2]  = 255;  // blue
			}
		}
	}

	system("cls");
	printf("%d", rand());
}