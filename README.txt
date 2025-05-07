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

# TODO 
1、需要确认garment unet不训练直接使用sd 1.5的unet (image和text propmt concat是否有影响)，是否会影响 vton unet的输出。 
2、logger的使用，在训练的时候能否正确输出重建图像，顺便熟悉推理的过程
3、image logger的可视化和以及图像的保存，确认正确
4、查看训练完42个epoch后的效果，并测指标，对比其它方法。





======================================================
模型下载
https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main
https://huggingface.co/openai/clip-vit-large-patch14/tree/main
https://huggingface.co/levihsu/OOTDiffusion/tree/main


每个 epoch 都意味着“完整地遍历一次训练集”。Batch size 不影响训练图像数量

batch size 越大 → 每个 epoch 的 steps 越少. steps = 数据集大小/ batch_size x epochs


====================================================================
当 do_classifier_free_guidance=True 时 输入两份 latent 到网络中：

    一份带 prompt（条件引导）

    一份不带 prompt（无条件引导

latent_model_input = torch.cat([latents] * 2) 

无条件（unconditional）预测（只输入噪声，不输入 prompt）,有条件（conditional）预测（输入噪声 + prompt）。
同时计算条件（有 prompt）和无条件（无 prompt）两种输入的预测值，以支持 classifier-free guidance
=====================================================================
num_images_per_prompt=4
在扩展生成时，它通常配合 prompt_embeds 使用。比如你输入一个 prompt：“a sunset over a mountain”，
如果你设置了 num_images_per_prompt=4，模型会为这个 prompt 生成 4 组 latent noise（随机初始点），分别反复引导生成出 4 张风格可能略有不同的图片。
初始 latent noise 是不同的，每张图都是从不同的随机起点出发
=======================================================================


------------------------------------------------------------------计划表
1、先统一 num_images_per_prompt看会不会报错，num_images_per_prompt = 4或者 =1，都试试 ✅ 
2、num_images_per_prompt =1时效果一样的情况下，全部去除encode，简写方法，看效果是不是一样 ✅ 
3、上述都完成的情况下，迁移到train训练代码的log_image中，看输出效果 ✅ 
4、查看seed 随机取值的逻辑，对生成有无影响
5、self.unet_garm self.unet_vton，可能受到prompt_embeds.shape和garm_latents.shape的影响, 打印下错误的shape是如何的 --->这个先做. ✅ 
6、确认正确的prompt_embeds.shape 和 garm_latents.shape是如何的 ✅ 
7、当所有训练和log_images是正常的之后，去学习prompt_embeds是如何注入到unet的，注入到unet的self attention 和 cross attention部分?
8、确认self.unet_garm输出这部分有什么影响，是否是固定的。我设置成固定的是否不影响最终效果
9、确认self.unet_garm输出 注入到self attention 和 cross attention哪个部分
10、确认IDM VTON的做法，对比是不是注入到self attention 和 cross attention一样的部分
11、把代码尽量与论文对应起来，学习理论知识
12、如何将prompt_embeds 对应成ip adapter和dinov2，并且将其训练。最好占用的显存不多。学习率能不能调高
13、确认创新点
14、编写论文

15、跑一下ootd和IDM VTON的指标，复现下确认下指标计算没有错误。论文先开始写，先投个会议
16、TODO 确认一下unet garment 注入 unet vton的是self-attention，image prompt注入unet ton的是 cross attention？
17、确认下模型是如何适配1024x768图像的, 经过缩放吗，sd 1.5是512分辨率

--------------------------------------------------------------------创新点
1、用dinov2去替代clip image propmt的图像特征和文本特征，试试效果,看看能不能替代
2、通道数原作者从4通道输入，扩展为了8通道，看一下我们能不能使用 vton+noise的方式,不使用vton concat noise的方式
3、使用vton+noise的方式 concat mask?看看效果有没有提升。TODOcheck是否需要加上姿态图等，我觉得没必要
4、class free guidance 不同取值做一个消融实验
5、添加一些增强模块等
6、确认下原来注入的garment的特征需要garment unet对应训练吧，可能不需要?
7、是否修改损失，但是我觉得这个比较难，没有必要
8、直接引用数据是一方面，但是可视化也要做到，所以还是需要复现

--------------------------------------------------------------------开始方法上的思考
1、anydoor的 condition只注入到 unet的解码器部分?
2、我能不能修改 IDM VTON的IP adapter为dinov2也作用到Unet的解码器部分?
3、看一下controlnet的封装和IDM VTON在原理上是不是一样，Self Attention和 Cross Attention 有没有区别
4、能不能把 dinov2的 + mask 二者结合作为condition注入到 Cross Attention
5、确认下anydoor的条件都是注入到Cross Attention吗，竟然效果还行吗，只是细节保留差点
6、验证下noise + vton是不是直接修改代码即可，可以看一下magic vton的noise + vton，学习模仿一下
7、其实ipadapter和controlnet是一样的引导作用，检查一下
8、确认controlnet 和 我们的代码在源码级别上添加的方式是不是一样的，可以学习magic clothing?
9、我看magic clothing中使用了controlnet的虚拟穿戴的细节效果已经达到了。查看源码进行模仿创新?

-------------------------------------------------------------------
1、使用ipadapter对image prompt进行编码注入到cross-attention中，看下是否能改善garment_unet冻结的效果
2、使用ipadapter我们做一下纯视觉的vton，作为创新点?
3、是不是可以不单单对garment进行编码，对mask和garment同时进行编码，能够更让garment适配
4、去想创新点，什么模块能拿来使用
5、对比验证效果，ootd冻结garment unet 22个epoch效果仍然差，原ootd是garment和vton一起进行训练的

备注：ControlNet显存占用很大，几乎 复制了一份 UNet 的结构（每层都有条件块）。IP-Adapter轻量，显存开销很小	只加了一些 cross-attn 层，没有复制 UNet
原来训练的模型别急着删除，试试效果

ablation study:1、不同的scale guidance，ipadapter对text, text + image，以及 image garment + mask等多种对比实验

-------------------------------------------------------------------
尝试几种训练情况：
1、都用clip的text encoder (好像效果区别并不大，需要从头开始训练做比较，建议直接跳过)
2、unet_garm使用clip的text encoder，unet_vton使用clip的text encoder + image encoder去finetune
3、unet_garm使用clip的text encoder(空)，unet_vton使用clip的text encoder(空) + image encoder去finetune
4、unet_garm和unet_vton 都使用 text encoder + image encoder，只 finetune unet_vton (**这种方案否决)
5、unet_garm和unet_vton 都使用 text encoder + image encoder，只 unet_garm 和 unet_vton一起finetune （论文的做法）
6、姿态图、轮廓图等，只需要加更多 key：added_cond_kwargs["pose_map"] 

---->TODO 衣服的文本提示词设置为["A cloth"], vton的提示词设置为["A model wears a cloth"], 不使用image encoder图像编码与text propmt结合, 并且使用ipadapter用于unet_vton
是否可以进一步使用controlnet,得到sobel纹理图输入，或者加一个模块可以做一些位置的矫正等等?

如果 UNet 是预训练 SD1.5，它的 cross-attn 是学来“理解文本 token”的；你硬塞一个图像 embedding，它其实不会“懂”它的含义。
prompt_embeds[:, 1:] = prompt_image[:], 把文本 embedding 中除了第一个 CLS token 的位置，都替换成图像编码向量, 所以文本不起作用?对应论文optional?
但是dress是concat的，有使用到文本信息

unet_garm to learn the garment features in a single step, 我可不可以多步去注入到unet_vton学习
---------------------------------------------------------------------

去检查一下unet_garm返回的是什么
跟随时间t返回spatial attention是不是好一点，看一下ootd和IDM的源码，都试试，尽量跟IDM VTON靠拢
比较下 ootd 和 IDM Vton，总不能sd 2.1和sd 1.5有区别吧?
基于sd 1.5好好改一下吧，在unet_garm冻结的情况下，看能不能复现效果，
感觉好像基于源码的和作者本来的，差了点效果


 "use_linear_projection": false  # TODO check true和false的区别，实在不行，跑一下IDM的工程测试下
 使用ipadapter估计需要看一下ipadapter的源码
 微调 UNet + 固定 IP-Adapter（常见）


 !!!!!!!!!!!!!!!!TODO 做一个对齐模块



 ==================================================
 要注意, vton和ootd训练效果出来不太一样,训练用vton，推理也用vton. 训练用ootd,推理也用ootd


 同时试试不同的文本图像嵌入，哪种组合对虚拟试穿的效果好，同时训练比如10个epoch试试

SD1.5 官方版	OpenAI/CLIP-ViT-L/14 （text encoder部分）
IP-Adapter（默认版，原版）	OpenAI/CLIP-ViT-L/14 （image encoder部分）
所以SD1.5和IP-Adapter用的版本是一样的

clip-vit-large-patch14 的 text encoder的输出是768维度的?图像编码器输出维度是1024,使用CLIPVisionModelWithProjection 会映射到768?

如果同时使用image encoder是不是没有必要，去除掉 image encoder充当文本注入到 unet，只使用ip adapter的image encoder, 
stablediffusionpipeline用的image encoder是IP adapter自带的?

# TODO 
1、首先保证迁移过来的源码推理出来的结果是与原来的代码一样的，然后再推理自己的看看结果
2、推理要和训练的prompt保持一致

# TODO check ootd 和 vton的prompt_embeds 是不一样的，注意看推理的时候有没有影响，训练和推理要确保一样
vton的推理:如果negative_prompt_embeds 是空的，好像是对""空字符串进行编码
ootd是将 两个prompt_embeds cat起来， 源码是negative_prompt_embeds和prompt_embeds相cat起来

----------------------------------------------------0430
unet_garm只用文本效果好像有点差，是不是unet_garm和unet_vton prompt一样比较好，试试纯文本试试
实验n个 epoch，以保留图像细节为主

============0504
用clip 的 text encoder和image encoder效果不好(原ootd)，试一下推理效果，对比ipadapter是否起到效果

============0505
caption = "A cloth"  和 caption = ""  要跟推理一致，看一下别人有没有进行修改，记得更改

============0506
使用ootd预训练unet_garm效果好了，去确认下为什么用预先训练的sd 1.5效果不好.

先推理看看mask如何使用的,再训练. 训练使用提前mask好的看看能不能提升训练速度. 看看提升多少


============0507
需要去看一下数据集读取顺序变了没有(数据集读取顺序没有变)
知道了原因sd1.5 不专注服装，注意力分布比较泛——人、背景、手势、脸、衣服都可能有注意力
看有没有办法在低显存的情况下进行unet_garm和unet_vton的联合训练

down_blocks\mid_block\upsample_block

Stable Diffusion 中 UNet 引入文本条件信息的关键模块。本质上是 Transformer 中的 Cross-Attention（交叉注意力）机制。始终启用： 只要你在生成时输入了 prompt_embeds（来自文本 prompt），这些 CrossAttention 就会参与工作。
Stable Diffusion 1.5 的 UNet 中就包含 Cross-Attn 模块。它们集中出现在：mid_block.attn1, up_blocks[i].attentions[j]（某些上采样块中）