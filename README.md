# CUDA-Circle-Renderer
Not just a software that draws circles on the screen, but also a tool which allows to benchmark GPU and CPU rendering times.

In the files "Spitaleri..." you will find explanatory PDFs (english and italian).

Build System:

This was made with Qt Creator on an Arch Linux laptop with dedicated NVidia chip.

I have installed the NVidia proprietary drivers, CUDA libraries, bumblebee package, and optirun package to able to make the graphic chip work only when needed (thank you NVidia for porting Optimus tecnology to Linux...).

This means that if you want to use your NVidia chip on your laptop you have to prepend "optirun" before the name of the built executable.

If you use Qt Creator this repo includes directives that tell Qt to prepend optirun when launching the application in its various options and modes. If you don't have a laptop with Optimus tecnology, but a tower PC with dedicated NVidia card you will not have to prepend the optirun keyword (so edit your project configuration accordingly).

In project settings, I pointed the Path to cuda toolkit to /opt/cuda. If yours is different be sure to edit that line accordingly.

Also note that only certain versions of CUDA libraries work only with certain versions of the NVidia drivers, as you might check here: https://docs.nvidia.com/deploy/cuda-compatibility/index.html . So, be sure not to update CUDA libraries if everything work fine.
