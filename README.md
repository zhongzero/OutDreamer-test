# OutDreamer



### Results



### Setup

```
conda create -n outdreamer python=3.8 -y
conda activate outdreamer
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.22.post7 --index-url https://download.pytorch.org/whl/cu118
pip install -e .
pip install -e ".[train]"
pip install -e ".[dev]"
```

### Download

* Pre-trained DiT model Weights  
  * We use pre-trained DiT model of [Open-Sora-Plan](https://github.com/Vchitect/Latte). To get the pre-trained DiT model weights, download them from the following link and put them into `pretrained_models/models--LanguageBind--Open-Sora-Plan-v1.2.0/29x720p` 
  * https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/29x720p
* VAE Weights
  * We use pre-trained VAE of [Open-Sora-Plan](https://github.com/Vchitect/Latte). To get the pre-trained VAE weights, download them from the following link and  put them into `pretrained_models/models--LanguageBind--Open-Sora-Plan-v1.2.0/vae` :
  * https://huggingface.co/LanguageBind/Open-Sora-Plan-v1.2.0/tree/main/vae

### train



### sample








### Acknowledgement
Our project is based on [Open-Sora-Plan](https://github.com/Vchitect/Latte) and we are grateful for its open-source nature.

