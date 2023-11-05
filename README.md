# Attentive Q-Matrix Learning for Knowledge Tracing
PyTorch implementation for QAKT.

This is the code for the paper:
[Attentive Q-Matrix Learning for Knowledge Tracing](https://arxiv.org/abs/2304.08168)  
Zhongfeng Jia, Wei Su, Jiamin Liu, Wenli Yue

This implementation is based on [AKT](https://arxiv.org/abs/2007.12324).

If you find this code useful in your research then please cite  
```bash
@misc{jia2023attentive,
      title={Attentive Q-Matrix Learning for Knowledge Tracing}, 
      author={Zhongfeng Jia and Wei Su and Jiamin Liu and Wenli Yue},
      year={2023},
      eprint={2304.08168},
      archivePrefix={arXiv},
      primaryClass={cs.CY}
}
```
## Setups
The requiring environment is listed in `environment.yaml`.


## Running QAKT.

The hyperparameters supported in this implementation can be found in `qakt_statics_pid_fold1-5.py`. Here is a simple example of how to use the QAKT model on statics:

```bash
python qakt_statics_pid_fold1-5.py --dataset statics_pid
```

