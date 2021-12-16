# EEGfusenet
Model architecture for the EEGfusenet and EEGCNNnet, and Discriminator
EEGfusenet_Channel_32: EEGfuseNet setting for 32 Channels signal (DEAP,HCI)
EEGfusenet_Channel_62: EEGfuseNet setting for 62 Channels signal (SEED)
Discriminator_Channel_32: Discriminator setting for 32 Channels signal (DEAP,HCI)
Discriminator_Channel_62: Discriminator setting for 62 Channels signal (SEED)


DEAP_model: Pretrained EEGfusenet model for DEAP dataset. The input of this model should be the preprocessed signal. Details about the preprocessing
procedures and the hypergraph clustering could be found in the appendix of our paper.
