Develop Log
===========
[Nov 1, 2013]
-----------
* After a set of 3-D data tested many times, I put those available functions into one file which named "cudaHelper.h". It will help the user to calculate the index of 3-D data, also provides macro functions for recording cudaError message.
* I also designed a number of functions, such as automatic calculation grid dimensions, block dimension.
* I don't understand why when I try to calculate the block dimension with third power of 64, vs prompts error message "stack overflow". 
* The last word, still bugs in code. 

[Oct 24, 2013]
-----------
* When I finished the 2-D CUDA version of CFD demo, I was starting to develop the 3-D version. First of all, I need to fixed the existing bugs in the 2-D.
* A new framework will be used for the new version.

[Oct 7, 2013]
-----------
* A simple CFD demo is now available, more features will be added on. 
* Structure of project still need to be optimized, in order to speed up.
* I will put this code on CUDA devices in the future.
* Currently is only 2D version, 3D will be soon on.

[Sep 27, 2013]
-----------
* Advect, diffuse, line solver methods without boundary conditions. In future, I need to add or modify those boundary conditions to avoid out of memory (array) run-time exception. 