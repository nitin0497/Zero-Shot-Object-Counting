# Zero-Shot-Object-Counting
This project reproduces the core components of the VA-Count framework, including the Exemplar Enhancement Module (EEM) for high-quality exemplar selection and the Noise Suppression Module (NSM) for exemplar-guided contrastive density estimation. 

<img width="2441" height="945" alt="image" src="https://github.com/user-attachments/assets/88f30d5a-fc22-4546-ad77-4230e45870d9" />

Below are more resources for replication and reuse. 

1) FSC147 dataset can be downloaded from here: https://drive.google.com/file/d/1ymDYrGs9DSRicfZbSCDiOu0ikGDh5k6S/view?usp=sharing
2) Precomputed ground truth density maps can be found here: https://archive.org/details/FSC147-GT
3) Pretrained model weights can be found here: https://drive.google.com/drive/folders/1jm-lYKTerqOEEg-A0zWRgdlj0paR4vXT?usp=sharing
   This folder also has all the required raw datasets, processed images, exemplars, and trained model weights.
4) The model training was monitored using Weights and Biases. Follow this link for visualisations and generated exemplars and density maps: https://api.wandb.ai/links/nitinyadav0497-auburn-university/uvl11odd

<img width="966" height="570" alt="image" src="https://github.com/user-attachments/assets/2bd61260-175f-4e32-a6a5-6e9118bcd38c" />

Illustration of the single object exemplar filtering with a frozen Clip-vit encoder and a trainable FFN to distinguish single from multiple objects.


<img width="975" height="338" alt="image" src="https://github.com/user-attachments/assets/d987b4b2-5275-4a17-a519-4d3a853002c7" />

Predicted vs. Ground-Truth Density Maps. White boxes show the top 3 exemplars of single object. 


Acknowledgement
This project is based on the implementation from 
