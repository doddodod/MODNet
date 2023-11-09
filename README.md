# information
fork from [MODNet with Dataset Repo](https://github.com/jarlyn95/MODNet) based on [MODNet Official Repo](https://github.com/ZHKKKe/MODNet) 
incorporate modules from P3M model [P3M Official Repo](https://github.com/ViTAE-Transformer/P3M-Net) 

Any query please check our git repositpry [Our Repo](https://github.com/doddodod/MODNet) 


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
1. Use the baseline: import from src.models.modnet_old. Modify the src/trainer.py to use line 151
2. Use the tfi: import from src.models.modnet_tfi. Modify the src/trainer.py to use line 151
3. Use the viTAE: import from src.models.modnet. Modify the src/trainer.py to use line 150

#Change dataset to the training dataset: change it in matting_dataset.py
#Change hyper-parameters: batch_size, total_epochs and learning_rate if needed. 
#trained model ".pth" will be save to pretrianed dir. 
#Same number of total_epochs images of result after each epoch will be save in dataset/UGD-12k/result, of image: test_image_paths. 

# 5. Pretrained Model: 
#Due to GitHub's capabilities, our pre-training model is saved in a Google Drive link.
#Save it in the pretrained folder if needed. 
#link: https://drive.google.com/drive/folders/1jxl29ine1rniXENvEKgiaJCUUHLHom2s?usp=drive_link 

# 6. Model evaluation
python eval.py
#Change dataset to the evaluation dataset: change it in eval.py line 148
#Use pretrained model to evaluate the dataset: change it in eval.py line 141
#Change ckp_pth to the pretrained model. 
#calculate the Mean Squared Error (MSE), Difference (MAD), Sum of Absolute Differences (SAD), Gradient (rate change of intensity in an image, magnitude of the gradient at each pixel), and Connectivity (how pixels in an image are connected or related to each other) between the groudthruth matte and the predicted alpha matte. 

# 7. Model Inference
python infer.py
#Ouput the predicted alpha matte by a given image, can select the input image and the pretrained model. 
```
