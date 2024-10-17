# LongPO: Long Context Self-Evolution of Large Language Models through Short-to-Long Preference Optimization

## Training Process:

1. Process the data, building the label for answer tokens and padding others.
  
2. Replace the Attention Module into Ulyssess Attn using monkey patch.
  
3. Replace the Trainer class into our custom Ulysses Trainer.
  
  1. LongPO Trainer: `LongDPOFullMTJointUlyssesTrainer`
    
  2. SFT Trainer using Ulysses: `LongSFTKLJointUlyssesTrainer`: Note that this Trainer uses our LongPO data format with a custom KL divergence. To access the naive SFT loss, refer to the chosen lm loss here.