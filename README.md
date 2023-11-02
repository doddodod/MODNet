# 说明
代码fork from [MODNet官方代码](https://github.com/ZHKKKe/MODNet) 。本项目完善了数据准备、模型评价及模型训练相关代码
# 模型训练、评价、推理
```bash
# 1. 下载代码并进入工作目录
git clone https://github.com/actboy/MODNet
cd MODNet

# 2. 安装依赖
pip install -r src/requirements.txt

# 3. 下载并解压数据集
#PPM-100: 
wget -c https://paddleseg.bj.bcebos.com/matting/datasets/PPM-100.zip -O dataset/PPM-100.zip
unzip dataset/PPM-100.zip 

#UGD-12k: 
https://github.com/fornorp/Interactive-Human-Matting/ 

# 4. 数据预处理
python matting_dataset.py
#Change the path in dataset_root_dir, to any training dataset. 
#Calculate trimpa by groundtruth matte and resize all images. 

# 5. 训练模型
python train.py
#Change batch_size, total_epochs and learning_rate if needed. 
#trained model ".pth" will be save to pretrianed dir. 
#Same number of total_epochs images of result after each epoch will be save in dataset/UGD-12k/result, of image: test_image_paths. 

# 6. 模型评估
python eval.py
#Change ckp_pth to the pretrained model. 
#Change dataset to the evaluation dataset. 
#calculate the Mean Absolute Difference (MAD) and Mean Squared Error (MSE) between the groudthruth matte and the predicted alpha matte. 

# 7. 模型推理
python infer.py
#Define the predicted alpha matte by given image and the pretrained model. 
```
