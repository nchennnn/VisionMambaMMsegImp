# VisionMambaMMsegImp
An **unofficial** implementation of VisionMamba(ViM) by mmsegmentation 0.30.

![ViM architecture](vim_pipeline_v1.9.png)

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

## Citation
Refer to plainmamba for mmseg style code, thanks!

```bibtex
 @article{vim,
  title={Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model},
  author={Lianghui Zhu and Bencheng Liao and Qian Zhang and Xinlong Wang and Wenyu Liu and Xinggang Wang},
  journal={arXiv preprint arXiv:2401.09417},
  year={2024}
}
```

```bibtex
@misc{yang2024plainmamba,
      title={PlainMamba: Improving Non-Hierarchical Mamba in Visual Recognition}, 
      author={Chenhongyi Yang and Zehui Chen and Miguel Espinosa and Linus Ericsson and Zhenyu Wang and Jiaming Liu and Elliot J. Crowley},
      year={2024},
      eprint={2403.17695},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
