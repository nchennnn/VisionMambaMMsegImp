# VisionMambaMMsegImp
An **unofficial** implementation of VisionMamba(ViM) by mmsegmentation 0.30.



## Envs
- CUDA11.7
- Pytorch 1.13
- mmsegmentation 0.30
- mamba-ssm 1.1.1

## Experimental TODO
- split xz proj into x+z proj
- compare the difference between bidirectional ssm and double unidirectional ssm (flip sequence or not)
- add a additional proj in front of (x+skip) skip connection
- add a middle cls token when training

