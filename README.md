# 基于扩散模型的注意力驱动RGB-D显著性目标检测
2025年发表于《信号处理》

# 网络架构
   <div align=center>
   <img src="https://github.com/Shixiang02/Attention-driven-RGB-D-salient-object-detection-based-on-diffusion-model/blob/main/image.png">
   </div>
   
   
# 环境配置
   python 3.10.16 + torch 2.5.1


# 性能表现
   本文推理的七个数据集显著性图可从此处下载：https://pan.baidu.com/s/1Q3Z3l3DIQlZqnA05-6CTZQ 提取码: XHCL
   12个对比模型的显著性图可从此处下载：https://pan.baidu.com/s/1fkAE1L9Wkd2bAAShdEYbcQ 提取码: XHCL
      
   ![Image](https://github.com/Shixiang02/Attention-driven-RGB-D-salient-object-detection-based-on-diffusion-model/blob/main/table1.png)
   ![Image](https://github.com/Shixiang02/Attention-driven-RGB-D-salient-object-detection-based-on-diffusion-model/blob/main/table2.png)
   
   
# 训练   
   下载主干网络预训练权重 [pvt_v2_b4_m.pth]:https://pan.baidu.com/s/1CAQeCbWHRx2ApheDkIByZw 提取码: XHCL, 放入 './pretrained_weights/'. 
   
   修改配置文件中数据集路径即可运行
   
   训练命令：
   accelerate launch train.py \
   --config config/camoDiffusion_352x352.yaml \
   --num_epoch=** \
   --batch_size=** \
   --gradient_accumulate_every=1



# 预训练权重及推理
  本文模型预训练权重（checkpoint）可从此处下载：https://pan.baidu.com/s/1PpUaPcvKQUuTF_v0nci2ew 提取码: XHCL
  
  推理命令：
  accelerate launch sample.py \
  --config config/camoDiffusion_352x352.yaml \
  --results_folder ** \
  --checkpoint ** \
  --num_sample_steps 10 \
  --target_dataset ** 

   
# 评价工具
   测评显著性图工具可从此处下载：https://github.com/lartpang/PySODEvalToolkit

   
感谢[Camodiffusion](https://github.com/Rapisurazurite/CamoDiffusion)对本工作的帮助




