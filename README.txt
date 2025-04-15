pip uninstall huggingface_hub
pip install huggingface_hub==0.25.2

pip install basicsr


# annotator_ckpts_path = os.path.join(PROJECT_ROOT, 'checkpoints/openpose/ckpts') #修改模型路径

# OOTDiffusion-Training/preprocess/humanparsing/run_parsing.py #修改模型路径

# 原作者的配置运行
At 512*384, it takes ~75 hours for 36000 steps with batch-size of 64 on a single A100 GPU (80 GB)

#48G显存对于我们不够用  batch size = 1也不行. 就算冻结一个unet, 最多设置batch size = 2. 没有太大意义，换！

# Magic clothing .  使用了冻结的sd unet, 可以试试， 只训练了garment unet


#训练的时候修改unet_path 的路径，看使用sd 1.5的unet，还是ootd的unet





======================================================
模型下载
https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main
https://huggingface.co/openai/clip-vit-large-patch14/tree/main
https://huggingface.co/levihsu/OOTDiffusion/tree/main


每个 epoch 都意味着“完整地遍历一次训练集”。Batch size 不影响训练图像数量

batch size 越大 → 每个 epoch 的 steps 越少. steps = 数据集大小/ batch_size x epochs