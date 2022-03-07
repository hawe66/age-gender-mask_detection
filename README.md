# Naver boostcamp: pstage_01_image_classification  
※ WARNING: image datasets are secured by copyright, thus all outputs showing images are erased!!  

## Experiment Description  
The task is to discriminate age/gender/mask from the face image.  
<p align="center">
  <a href="#task-summary"><img width="100%" src="https://github.com/hawe66/age-gender-mask_detection/tree/main/imgs/56bd7d05-4eb8-4e3e-884d-18bd74dc4864.png" /></a>
</p>
<p align="center">
  <a href="#EDA"><img width="100%" src="https://github.com/hawe66/age-gender-mask_detection/tree/main/imgs/56bd7d05-4eb8-4e3e-884d-18bd74dc4864.png" /></a>
</p>
  
From these description, you can guess the main objective of this task is to handle the unbalanced labels.  
  - Age >= 60 are only 7 % of the whole data  
  - Unmasked, Incorrect are only 14 % of the whole data, respectively  
  
**My approach was to solve this problem via...**  
  1) Using **resnet18** (relatively slight model) as a pretrained model and fine-tune  
  2) Using **[smote algorithm](https://github.com/ufoym/imbalanced-dataset-sampler)** to oversample the sparse data  
  3) Using **weighted focal loss** to signify the sparse label detection  
  4) Using **Test Time Augmentation (TTA)** to get robust decision from a model  
  
  additional tries...
  + center crop & random crop while remaining **original pixel sizes(fatal!)**  
  + Separating tasks (age/gender/mask) and encode multi labels from the three separate models  
  + Using model weights pretrained from age detection task  

  
## Getting Started    
  
### Workflow  
- Training: `python train.py --name your_experiment`   
  - for more options, try `python train.py -h`  
  - for splitting tasks into 3 discrete tasks, `python train_splitted.py --name your_age_experiment`  
- Inference: `python inference.py`  
  - for using TTA (test time augmentation), `python inference_TTA.py`  
  - for splitted, `python inference_ensemble.py`  
- models saved at `model`  
- additional validation saved at `exploration`  
  - EDA, Inference Validation, ... are done by ipython kernel  
- model weights pretrained by face detection task stored at `pretrained`  
  - links, methods are described in the [pretrained](https://github.com/hawe66/age-gender-mask_detection/tree/main/pretrained)  
  
## Settings  
  
### Dependencies
- torch==1.7.1
- torchvision==0.8.2                                                              

### Install Requirements
- `pip install -r requirements.txt`

### Training
- `SM_CHANNEL_TRAIN={YOUR_TRAIN_IMG_DIR} SM_MODEL_DIR={YOUR_MODEL_SAVING_DIR} python train.py`

### Inference
- `SM_CHANNEL_EVAL={YOUR_EVAL_DIR} SM_CHANNEL_MODEL={YOUR_TRAINED_MODEL_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python inference.py`

### Evaluation
- `SM_GROUND_TRUTH_DIR={YOUR_GT_DIR} SM_OUTPUT_DATA_DIR={YOUR_INFERENCE_OUTPUT_DIR} python evaluation.py`
