#include "Headers.h"
#include "Visualization.h"
#include <time.h>

using namespace sge;

_volumeData            vbdata;
extern Visualization   *visual;

DWORD simulation (LPVOID lpdwThreadParam )
{
#ifdef _USING_3D_TEXTURE
	vbdata.width = 256;
	vbdata.height = 256;
	vbdata.depth = 256;
	vbdata.textureID = 0;

	void update();

	vbdata.data = (unsigned char*) malloc (
		sizeof(unsigned char) * vbdata.width * vbdata.height * vbdata.depth * BYTES_PER_TEXEL);
#endif
#ifdef _USING_2D_TEXTURE
	vbdata.width = 256;
	vbdata.height = 256;
	vbdata.textureID = 0;

	void update();

	vbdata.data = (unsigned char*) malloc (
		sizeof(unsigned char) * vbdata.width * vbdata.height * BYTES_PER_TEXEL);
#endif

	srand(time(NULL));

	while( TRUE )
	{
		visual->LoadVolumeData(&vbdata);
		update();
	}
};

void update()
{
#ifdef _USING_3D_TEXTURE
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
#endif
#ifdef _USING_2D_TEXTURE
	for (int i=0; i<vbdata.width; i++) {
		for (int j=0; j<vbdata.height; j++) {
				vbdata.data[visual->TEXEL2(i, j)]  = 255;  // red
				vbdata.data[visual->TEXEL2(i, j) + 1]  = 0;  // green
				vbdata.data[visual->TEXEL2(i, j) + 2]  = 255;  // blue
		}
	}
#endif
}