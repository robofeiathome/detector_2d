# detector_2D

## Faz detecção a leitura do tópico da zed2, e publica a imagem com labels no tópico <i>```detector_2d/objects_label/image```</i>.

**Requirements to run on jetson using GPU**
- torch==1.14.0a0+44dac51c.nv23.02
- torchvision==0.15.0a0+0e62c34

First you must have to uninstall the current torch/torchvision version. The documentation to install the torch [here](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) and the torchvision can not be installed by pip, **have to be installed by [source](https://github.com/pytorch/vision)**, and to work with te current torch (jetson) version correctly, before you run the setup.py install, you have to replace two directories that you can find [here](https://github.com/cauansousa/torchvision-to-jetson) (follow the replace steps in the readme), and then run ```pythonX.X setup.py install --user``` in the vision dir.
