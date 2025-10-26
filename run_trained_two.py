import copy

import torch
from diffusers import DiffusionPipeline
import torch
from architect_modify import sd3_adapter_attn_processor
from image_process import ImageProjModel
from PIL import Image
import torch
from torchvision import transforms
from time import time
import torch.nn.functional as F
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
import numpy as np
import random
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
dtype=torch.float16

controlnet = SD3ControlNetModel.from_pretrained("controlnet/", torch_dtype=torch.float16)

# pipe=StableDiffusion3ControlNetPipeline.from_pretrained("pre_models/",controlnet=controlnet, torch_dtype=dtype).to("cuda")

#加载权重 为预训练文件加载权重
pipe = DiffusionPipeline.from_pretrained("pre_models/", torch_dtype=dtype).to("cuda")

from transformers import CLIPVisionModelWithProjection

# clip vision
image_encoder = CLIPVisionModelWithProjection.from_pretrained("D:\\python_file\\IP-Adapter\\CLIP",
                                                              torch_dtype=dtype).to("cuda")

text_tokenizer = CLIPTokenizer.from_pretrained("D:\\python_file\\IP-Adapter\\CLIP",torch_dtype=dtype)
text_encoder = CLIPTextModelWithProjection.from_pretrained("D:\\python_file\\IP-Adapter\\CLIP",torch_dtype=dtype).cuda()

# print(image_encoder)
def pics_process(transform,image_encoder,image_proj,text_features,vae,img=None,canny_pic=None):

    img1=img
    # img1 = Image.open(pic_path)
    if img1.mode == "RGBA":
        img1 = img1.convert('RGB')

    heat_ori_pic=img1

    img1 = transform(img1)
    img1=img1.to("cuda")
    img1 = img1.unsqueeze(0).to(torch.float16)
    img1_embed = image_encoder(img1)
    #正常VGEN
    # img1_embed = img1_embed.last_hidden_state.to(torch.float16)  # [batch_size,257,1280]
    # last_hidden_state1 = img1_embed[:, 1:, :]
    # embed_concat, score_clip = image_proj(last_hidden_state1, text_features,heat_ori_pic)
    #只使用最后一层
    last_hidden_state = img1_embed.image_embeds.to(torch.float16)  # [batch_size,257,1280]
    embed_concat, score_clip = image_proj(last_hidden_state, text_features)

    return embed_concat,score_clip




seed = 42


adapter=True

if adapter:
    vae=pipe.vae
    #获取transformer模块
    pipe_transformer=pipe.transformer
    attn_procs={}
    # 替换注意力处理模块
    for name in pipe_transformer.attn_processors.keys():
        layer_name = name.split(".processor")[0]

        attn_procs[name] = sd3_adapter_attn_processor().to(torch.float16)

    #注入注意力处理模块
    pipe_transformer.set_attn_processor(attn_procs)
    #获取预训练权重字典
    dict=pipe_transformer.state_dict()
    #加载训练权重
    trained_dict=torch.load("./save_model/last-600000.pth",map_location="cuda",weights_only=True)


    # 'image_proj_model_dict'
    # 'weights_to_save'
    image_proj_dict = trained_dict['image_proj_model_dict']
    other_weights = trained_dict['weights_to_save']

    # total_params_a = sum(p.numel() for p in other_weights.values())
    # total_params_b = sum(p.numel() for p in image_proj_dict.values())
    #
    # print(f"预训练模型的参数总数: {total_params_a+total_params_b}")


    image_proj = ImageProjModel().to("cuda")
    image_proj.load_state_dict(image_proj_dict)

    #权重替换
    for name in dict.keys():

        #"module."+

        if "sd3_to_" in name:
            weights = other_weights[name].to(dtype)
            dict[name] = weights

    #transformer加载权重
    pipe_transformer.load_state_dict(dict)
    pipe_transformer = pipe_transformer.to("cuda")
    pipe.transformer = pipe_transformer

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])

    # Impressionism, modern art, oil painting, watercolor
    # wear glasses ,wear sunglasses,angry face,Traditional Chinese clothing
    import cv2
    prompt = ("a yellow bus,a blue car")
    text_inputs = text_tokenizer(prompt, padding=True, return_tensors="pt").to("cuda")

    # 得到 text embedding（就是 CLIP 的语义向量)
    old=True
    text_features = text_encoder(**text_inputs)
    if old:
        text_features=text_features.text_embeds
    else:
        text_features=text_features.last_hidden_state

    # 1. 读取图片
    img_path = './test_img/000000007339.jpg'
    img = Image.open(img_path)
    width, height = 768, 768

    def resize_and_crop(image, target_size=512, noise_std=0.01):
        # 获取原始图像尺寸
        width, height = image.size

        # 计算等比缩放的比例，使短边缩放到512
        if width < height:
            scale = target_size / width
        else:
            scale = target_size / height

        new_width = int(width * scale)
        new_height = int(height * scale)

        # 使用新的尺寸缩放图像
        image_resized = image.resize((new_width, new_height), Image.BILINEAR)

        # 进行中心裁剪，使图像大小为512x512
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = (new_width + target_size) // 2
        bottom = (new_height + target_size) // 2

        image_cropped = image_resized.crop((left, top, right, bottom))

        return image_cropped


    # img=resize_and_crop(img)


    output,scores=pics_process(transform,image_encoder,image_proj,text_features=text_features,vae=vae,img=img)

    # VGEN使用CLIP最后一层




    guidance_scale = 7.5
    if guidance_scale > 0:
        img_embed=torch.cat((output,output),dim=0)

    import random

    for i in range(3):

        seed=random.randint(0, 1000000000)

        print(seed)

        generator = torch.Generator(device="cuda").manual_seed(seed)
        image1 = pipe(
            prompt=prompt,
            negative_prompt="texts,blurry, low quality, bad anatomy, disfigured, deformed, extra limbs, missing fingers, fused fingers, bad proportions, out of frame",
            num_inference_steps=30,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            # seed=seed,
            generator=generator,
            image_embed=[img_embed,scores],
            origin_image=img
        ).images[0]

        import cv2, numpy as np

        arr = np.array(image1.convert("RGB"))
        cv2.imwrite(f"./output/{time()}.jpeg", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


