## Pytorch_AnormalyDetection
+ Pytorch
+ Anomaly Detection
+ Python

## Version
+ Python : 3.7.6
+ torch : 1.8.0+cu111
+ torchvision : 0.9.0+cu111
+ cv2 : 4.4.0
+ skimage : 0.14.2
+ matplotlib : 3.0.3
+ numpy : 1.18.2

## Usage Example
### Dataset
MvTEC Anormaly detection dataset.  
Link : https://www.mvtec.com/company/research/datasets/mvtec-ad/

### 1. Open scripts/main.py file  
![1](https://user-images.githubusercontent.com/20108771/126437924-9d5a32da-dd52-480c-a7d8-24cffb47e667.PNG)

### 2. Set Parameter for Train, Test  
![2](https://user-images.githubusercontent.com/20108771/126438024-912c2b95-4c8c-4477-ac4c-8cea585cf485.PNG)

+ learning_rate : optimizer learning late 
+ num_epoch : train epoch
+ batch_size : batch size
+ data_dir : train data directory
+ test_dir : test data directory
+ backup_dir : save model directory
+ train_loop : True = Run train, test, False = only test
+ use_ssim : True = use ssim, False = just diff (cv2.absdiff())
+ diff_thresh : diff threshold value

### 3. Run main.py
![3](https://user-images.githubusercontent.com/20108771/126442845-019ab737-edf0-4905-8b1f-22fe1921dee8.PNG)

### 4. Result
* use ssim  
  
![4](https://user-images.githubusercontent.com/20108771/126443246-3f3a8e63-058c-4687-b08e-bffd13591bdd.png)

* just cv2.absdiff  
  
![4](https://user-images.githubusercontent.com/20108771/126443071-5644bc39-859a-4864-837b-28f2fe46119f.png)
  
  
  
