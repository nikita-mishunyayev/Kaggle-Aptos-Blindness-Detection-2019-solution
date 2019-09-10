## Aptos Blindness Detection 2019

Our team ranked 82nd place (TOP 3%) in the [Aptos Blindness Detection 2019 on Kaggle platform](https://www.kaggle.com/c/aptos2019-blindness-detection/leaderboard). This repository consists of code and configs that I used to train one of our best single model. The solution is powered by awesome [Catalyst](https://github.com/catalyst-team/catalyst) and [Albumentations](https://github.com/albu/albumentations) libraries.

## In this repository you can find:
* `training.ipynb` - simple notebook with three stages of **EfficientNet(b4)** training. The first stage is pre-training on the data of 2015. The second is training at 80% of the 2019 data and choosing the best epoch number on validation. Third, training on all 2019 data. This single model gave **0.801 on Public LB (15%) and 0.917 on Private LB (85%)**
* `inference.ipynb` - notebook with simple inference
* `src` - folder with basic scripts for learning models
* `configs` - folder with pretraining and finetuning configs

## Solution description

### Data
2015 competition data was used for pretraining all our models. Without it out models performed much worse. We used different techniques: first train on old data, then finetuning on the new train, another technique train on both data, the finetune on new train data. Besides, starting finetuning with freezing all layers and training only last FC layer gave us more stable results.

We tried many neural network architectures in this competition, but we achieved our two best results by blending two models: rnn and rcnn. It was an interesting competition, primarily because I had never used recurrent and convolutional neural networks before. Also, the public leaderboard is calculated with approximately 15% of the test data and private with 85%, which led to a large shake-up and to the fact that different seed gave different results, even on a private leaderboard. We flew far down, but it was a great experience, from which I learned a lot of new things.

### Models
- From the beginning, efficientnet outperformed other models. Using fp16 (available in kaggle kernels) allowed to use bigger batch size - speeded up training and inference. Other models (like ResNet, ResNext or DenseNet) worked worse for us.
- Models used in the final submission:
  1. EfficientNet-B5 (best single model): 224x224 (tta with Hflip, preprocessing - crop_from_gray, circle_crop, ben_preprocess=10)
  2. EfficientNet-B4: 256x256 (tta with Hflip, preprocessing - crop_from_gray, circle_crop, ben_preprocess=20)
  3. EfficientNet-B5: 256x256 (tta with Hflip, preprocessing - crop_from_gray, circle_crop, ben_preprocess=30)
  4. EfficientNet-B5: (256x256) without specific preprocess, two models with different augmentations.
- We tried bigger image sizes but it gave worse results. EfficientNet-B2 and EfficientNet-B6 gave worse results as well.

### Augmentations
From [Albumentations](https://github.com/albu/albumentations) library:
Hflip, VFlip,  RandomScale, CenterCrop,  RandomBrightnessContrast, ShiftScaleRotate, RandomGamma, RandomGamma, JpegCompression, HueSaturationValue, RGBShift, ChannelShuffle, ToGray, Cutout

### Training
First 3 models were trained using [Catalyst](https://github.com/catalyst-team/catalyst) library and the last one with FastAi, both of them work on top of Pytorch.

We used both ordinal regression and regression. Models with classification tasks weren't well enough to use them.
Adam with OneCycle was used for training. WarmUp helped to get more stable results. RAdam, Label smoothing didn't help to improve the score.

We tried to use leak investigated [here](https://www.kaggle.com/miklgr500/leakage-detection-about-8-test-dataset) and [here](https://www.kaggle.com/konradb/adversarial-validation-quick-fast-ai-approach) by fixing output results. Almost 10% of the public test data were part of the train. Results dropped significantly, which means training data annotation were pretty bad.

We tried kappa coefficient optimization, it didn't give reliable improvement on public, but could help us on private almost +0.003 score.

### Hardware
We used 1x* *2080*, 1x* *Tesla v40*, 1x* *1070ti*, kaggle kernels

## Team:
- Mamat Shamshiev: [Kaggle](https://www.kaggle.com/mamatml), [GitHub](https://github.com/MamatShamshiev)
- Insaf Ashrapov: [Kaggle](https://www.kaggle.com/insaff), [GitHub](https://github.com/Diyago)
- Mishunyayev Nikita: [Kaggle](https://www.kaggle.com/mnikita), [GitHub](https://github.com/Mishunyayev-Nikita)
