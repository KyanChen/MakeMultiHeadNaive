# MakeMultiHeadNaive
Use naive MultiheadAttention implement to replace nn.MultiheadAttention in pytorch

If you find this project helpful, please give us a star ⭐️, your support is our greatest motivation.

本代码使用朴素的线性层来替换Pytorch中的[多头注意力](https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html)，这使得基于torch.nn.MultiheadAttention实现的Transformer(比如[CLIP、OpenClip](https://github.com/mlfoundations/open_clip))也可以使用Hugingface的PEFT(例如[LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora))进行微调。

The code uses a simple Linear layer to replace the nn.MultiheadAttention in pytorch, making the Transformers (such as [CLIP、OpenClip](https://github.com/mlfoundations/open_clip)) based on torch.nn.MultiheadAttention fine-tuning with Hugingface's PEFT (such as [LoRA](https://huggingface.co/docs/peft/conceptual_guides/lora)).
