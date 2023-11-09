# information
fork from [MODNet with Dataset Repo](https://github.com/jarlyn95/MODNet) based on [MODNet Official Repo](https://github.com/ZHKKKe/MODNet) 
incorporate modules from P3M model [P3M Official Repo](https://github.com/ViTAE-Transformer/P3M-Net) 
# training ,infrence and evaluation
```bash
# 1. download the code
git clone https://github.com/doddodod/MODNet.git
cd MODNet

# 2. install dependencies
pip install -r src/requirements.txt

# 3. dowload dataset with gdown and google drive link
#PPM-100: 
gdown --id 1JUx-EPoV9QAhQgmW0AyOen-xKQUzZia- --output dataset/ppm-100.zip
unzip dataset/PPM-100.zip -d PPM-100

#UGD-12k: 
https://github.com/fornorp/Interactive-Human-Matting/ 
#data should be save in dataset, create a floder "train", and put folder image and alpha of train dataset of UGD-12k in it.
#also create a floder "eval", and put folder image and alpha of train dataset of UGD-12k in it.
    
mkdir dataset/UGD-12k
#train
# link: https://drive.google.com/file/d/1xeHNIXUl4NTuMcwnDSCfail_IixnG3UZ/view
gdown --id 1xeHNIXUl4NTuMcwnDSCfail_IixnG3UZ --output dataset/UGD-12k/train.zip
! unzip dataset/UGD-12k/train.zip -d train
#evaluation
# link: https://drive.google.com/file/d/1a3uyv4Ce-N_9CHGTFNV3qMJMZ0Qb4zSi/view
gdown --id 1a3uyv4Ce-N_9CHGTFNV3qMJMZ0Qb4zSi --output dataset/UGD-12k/eval.zip
! unzip dataset/UGD-12k/eval.zip -d eval
#test
# link: https://drive.google.com/file/d/1UZTbEFv5KhOrajthf_jGuKAmxxjbgpoj/view
gdown --id 1UZTbEFv5KhOrajthf_jGuKAmxxjbgpoj --output dataset//UGD-12k/test.zip
unzip dataset/UGD-12k//test.zip -d test

# p3m-10k
gdown --id 1LqUU7BZeiq8I3i5KxApdOJ2haXm-cEv1 --output dataset/p3m.zip
unzip dataset/p3m.zip -d p3m-10k


# 4. Model training
python train.py

#Change Model in train.py:
1. Use the baseline: import from src/models/modnet_old
2. Use the tfi: import from src/models/modnet_tfi
3. Use the viTAE: import from src/models/modnet

#Change dataset to the training dataset.
#Change hyper-parameters: batch_size, total_epochs and learning_rate if needed. 
#trained model ".pth" will be save to pretrianed dir. 
#Same number of total_epochs images of result after each epoch will be save in dataset/UGD-12k/result, of image: test_image_paths. 

# 5. Model evaluation
python eval.py
#Import model corresponding to pretrained model, same as in 4. 
#Change ckp_pth to the pretrained model. 
#Change dataset to the evaluation dataset. 
#calculate the Mean Absolute Difference (MAD) and Mean Squared Error (MSE) between the groudthruth matte and the predicted alpha matte. 

# 6. Model Inference
python infer.py
#Ouput the predicted alpha matte by given image, model imported and selected pretrained model. 
```
