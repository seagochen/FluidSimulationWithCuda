#include "Visual.h"
#include <time.h>

using namespace sge;

extern Visual *visual;

_volume2D vbdata;


DWORD simulation (LPVOID lpdwThreadParam )
{
	vbdata.width = 256;
	vbdata.height = 256;
	vbdata.texture_id;

	void update();

	vbdata.data = (unsigned char*) malloc (
		sizeof(unsigned char) * vbdata.width * vbdata.height * BYTES_PER_TEXEL);

	srand(time(NULL));

	while( TRUE )
	{
		visual->UploadVolumeData(&vbdata);
		update();
	}
};

void update()
{
	for (int i=0; i<vbdata.width; i++) 
	{
		for (int j=0; j<vbdata.height; j++) 
		{
			vbdata.data[visual->Volume2D(i, j)]  = 255;  // red
			vbdata.data[visual->Volume2D(i, j) + 1]  = 0;  // green
			vbdata.data[visual->Volume2D(i, j) + 2]  = 255;  // blue
		}
	}
}