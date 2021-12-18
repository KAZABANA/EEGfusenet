# EEGfusenet
Model architecture for the EEGfusenet and EEGCNNnet, and Discriminator
EEGfusenet_Channel_32: EEGfuseNet setting for 32 Channels signal (DEAP,HCI)
EEGfusenet_Channel_62: EEGfuseNet setting for 62 Channels signal (SEED)
Discriminator_Channel_32: Discriminator setting for 32 Channels signal (DEAP,HCI)
Discriminator_Channel_62: Discriminator setting for 62 Channels signal (SEED)

Pretrained_model: Pretrained EEGfusenet model for DEAP dataset, you could load it by function torch.load.
demo.py: Simple demo code for deep feature extraction based on our pretrained model.

The input of this model should be the preprocessed signal. Details about the preprocessing procedures and hypergraph clustering can be found in the appendix of our paper.
Appendix Link: https://drive.google.com/file/d/1pIphu7LD5MrHsN6GRy1-xZcc9CtV24tm/view?usp=sharing



If you find the paper or this repo useful, please cite:
@article{9535130,  author={Liang, Zhen and Zhou, Rushuang and Zhang, Li and Li, Linling and Huang, Gan and Zhang, Zhiguo and Ishii, Shin},  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},   title={EEGFuseNet: Hybrid Unsupervised Deep Feature Characterization and Fusion for High-Dimensional EEG With an Application to Emotion Recognition},   year={2021},  volume={29},  number={},  pages={1913-1925},  doi={10.1109/TNSRE.2021.3111689}}
