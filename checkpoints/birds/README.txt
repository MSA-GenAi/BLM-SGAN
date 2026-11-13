One folder per model. Each folder holds:
  netG_<epoch>.pth   the generator checkpoint (the highest epoch in the folder is used)
  model.json         {"name": ..., "kind": "blm" | "ssagan", "order": <number>, "description": ...}
  text_encoder.pth   kind "blm" only: the BERT-DAMSM text encoder the generator was trained with
kind "blm"    = BLM-SGAN (training/); create the folder with training/export_model.py, or download the release files
kind "ssagan" = SSA-GAN, conditioned on AttnGAN's DAMSM LSTM encoder (needs DAMSMencoders/bird/text_encoder200.pth)
