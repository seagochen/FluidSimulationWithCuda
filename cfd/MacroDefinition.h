#ifndef _DEFINITION_H_
#define _DEFINITION_H_

#define SpaceToTexel(x, y) visual->Texel2D(y, x)
#define CleanDirty() { vbdata.data[SpaceToTexel(i, j)] = 0; vbdata.data[SpaceToTexel(i, j) + 1] = 0; vbdata.data[SpaceToTexel(i, j) + 2]  = 0; }

#endif