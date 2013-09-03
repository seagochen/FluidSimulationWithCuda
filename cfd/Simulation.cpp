#include "Headers.h"
#include "Visualization.h"

using namespace sge;

_volumeData            vbdata;
extern Visualization   *visual;

DWORD simulation (LPVOID lpdwThreadParam )
{
	vbdata.width = 128;
	vbdata.height = 128;
	vbdata.depth = 128;
	vbdata.textureID = 0;

	void update();

	vbdata.data = (unsigned char*) malloc (
		sizeof(unsigned char) * vbdata.width * vbdata.height * vbdata.depth * BYTES_PER_TEXEL);

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
				vbdata.data[visual->TEXEL3(i, j, k)] = i % 255;
				vbdata.data[visual->TEXEL3(i, j, k) + 1] = j % 255;
				vbdata.data[visual->TEXEL3(i, j, k) + 2] = k % 255;
			} // for
		} // for
	} // for
}