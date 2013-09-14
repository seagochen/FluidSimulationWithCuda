#include "Visual.h"
#include <time.h>

using namespace sge;
extern Visual *visual;
_volume2D vbdata;

#include "MacroDefinition.h"

DWORD simulation (LPVOID lpdwThreadParam )
{
	vbdata.width = 256;
	vbdata.height = 256;
	vbdata.texture_id;
	vbdata.size = sizeof(GLubyte) * vbdata.width * vbdata.height * BYTES_PER_TEXEL;

	void update();

	vbdata.data = (GLubyte*) malloc ( vbdata.size );

	srand(time(NULL));

	// whether visualization is ready for rendering result
	while( TRUE )
	{
		visual->UploadVolumeData(&vbdata);
		update();
	}

	return 0;
};


void update(void)
{
	for (int i=0; i<vbdata.width; i++)
	{
		for (int j=0; j<vbdata.height; j++)
		{
			// red
			if (i<60 && j<60)
			{
				vbdata.data[SpaceToTexel(i, j)]  = 255;      // red
				vbdata.data[SpaceToTexel(i, j) + 1]  = 0;    // green
				vbdata.data[SpaceToTexel(i, j) + 2]  = 0;    // blue
			}

			// yellow
			else if (i>vbdata.width - 60 && j >vbdata.height - 60)
			{
				vbdata.data[SpaceToTexel(i, j)]  = 255;      // red
				vbdata.data[SpaceToTexel(i, j) + 1]  = 255;    // green
				vbdata.data[SpaceToTexel(i, j) + 2]  = 0;    // blue
			}

			// blue
			else if (i>vbdata.width - 60 && j<60)
			{
				vbdata.data[SpaceToTexel(i, j)]  = 0;      // red
				vbdata.data[SpaceToTexel(i, j) + 1]  = 0;    // green
				vbdata.data[SpaceToTexel(i, j) + 2]  = 255;    // blue
			}

			// green
			else if (i<60 && j >vbdata.height - 60)
			{
				vbdata.data[SpaceToTexel(i, j)]  = 0;      // red
				vbdata.data[SpaceToTexel(i, j) + 1]  = 255;    // green
				vbdata.data[SpaceToTexel(i, j) + 2]  = 0;    // blue
			}

			else { CleanDirty(); }
		}
	}
}