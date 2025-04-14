pip uninstall huggingface_hub
pip install huggingface_hub==0.25.2

pip install basicsr


# annotator_ckpts_path = os.path.join(PROJECT_ROOT, 'checkpoints/openpose/ckpts') #修改路径

# 原作者的配置运行
At 512*384, it takes ~75 hours for 36000 steps with batch-size of 64 on a single A100 GPU (80 GB)

#48G显存对于我们不够用  batch size = 1也不行. 就算冻结一个unet, 最多设置batch size = 2. 没有太大意义，换！


#训练的时候修改unet_path 的路径，看使用sd 1.5的unet，还是ootd的unet