<div align=center> <h1>
  <img align="left" width="200" height="170" src="assets/LogoWithoutBG.png" alt="MicroUS and Histopathology Registration">
  Image Registration of <i>In Vivo</i> Micro-Ultrasound and <i>Ex Vivo</i> Pseudo-Whole Mount Histopathology Images of the Prostate: A Proof-of-Concept Study</h1>

Welcome to the repository containing the code for our **Microu-Us and Histopathology Image Registration Newtork**, a deep learning model designed for registering micro-ultrasound and histopathology images.
  
[![](https://img.shields.io/badge/Imran-gray?logo=github&logoColor=white&label=Muhammad&labelColor=darkgreen&color=red)](https://www.linkedin.com/in/imrannust/) &emsp;
[![](https://img.shields.io/badge/Nguyen-gray?logo=ResearchGate&logoColor=white&label=Brianna&labelColor=darkblue&color=limegreen)](https://www.researchgate.net/profile/Brianna_Nguyen2) &emsp;
[![](https://img.shields.io/badge/Pensa-gray?logo=linkedin&logoColor=white&label=Jake&labelColor=black&color=yellow)](https://www.researchgate.net/profile/Brianna_Nguyen2) &emsp;
[![](https://img.shields.io/badge/Falzarano-gray?logo=linkedin&logoColor=white&label=Sara&labelColor=darkred&color=cyan)](https://www.linkedin.com/in/sara-falzarano-3a788941/) &emsp;
[![](https://img.shields.io/badge/Sisk-gray?logo=world%20health%20organization&logoColor=white&label=Anthony&labelColor=darkgreen&color=orange)](https://www.uclahealth.org/providers/anthony-sisk) &emsp;
[![](https://img.shields.io/badge/Liang-gray?logo=linkedin&logoColor=white&label=Muxuan&labelColor=darkpurple&color=lime)](https://www.linkedin.com/in/muxuan-liang-5b98aa47/) &emsp;
[![](https://img.shields.io/badge/DiBianco-gray?logo=world%20health%20organization&logoColor=white&label=John%20Michael&labelColor=darkslategray&color=fuchsia)](https://urology.ufl.edu/about-us-2/faculty-staff-directory-3/john-michael-dibianco-md/) &emsp;
[![](https://img.shields.io/badge/Su-gray?logo=world%20health%20organization&logoColor=white&label=Li-Ming&labelColor=darkolivegreen&color=purple)](https://urology.ufl.edu/about-us-2/meet-our-team/li-ming-su-md/) &emsp;
[![](https://img.shields.io/badge/Zhou-gray?logo=github&logoColor=white&label=Yuyin&labelColor=darkorange&color=blue)](https://yuyinzhou.github.io/) &emsp;
[![](https://img.shields.io/badge/Joseph-gray?logo=linkedin&logoColor=white&label=Jason%20P.&labelColor=navy&color=orange)](https://urology.ufl.edu/about-us-2/faculty-staff-directory-3/jason-p-joseph-md/) &emsp;
[![](https://img.shields.io/badge/Brisbane-gray?logo=world%20health%20organization&logoColor=white&label=Wayne%20G.&labelColor=darkcyan&color=magenta)](https://www.uclahealth.org/providers/wayne-brisbane) &emsp;
[![](https://img.shields.io/badge/Shao-gray?logo=linkedin&logoColor=white&label=Wei&labelColor=darkviolet&color=teal)](https://www.linkedin.com/in/wei-shao-438782115/)

</div>

## Repository Contents



1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Dataset Detail](#dataset-detail)
4. [Directory Structure](#directory-structure)
5. [Scripted Code](#scripted_code)
6. [Interactive Code](#interactive_cde)
7. [Dependencies](#dependencies)
8. [Citations](#citations)

---

<h2>Overview <a id="overview" ></a></h2>

---

<h2>Directory Structure<a id=directory-structure"></a></h2>

Let's clone the repository as follows:

<h3> 1. Clone the Repository:</h3>

  Open your terminal or command prompt and clone the project directory as follows:
  ```
  git clone https://github.com/mirthAI/MUS-Pathology-Registration.git
  ```

<h3> 2. Navigate to the Directory: </h3>

  Once the repository is cloned, navigate to the desired directory using the `cd` command as follows:
  ```
  cd MUS-Pathology-Registration
  ```
<h3> 3. Directory Structure </h3>


```
MUS-Pathology-Registration/
├── data/                                                  # Main data directory in NIfTI format
│   ├── Subject002/
│   │   ├── microUS_3D_Subject002_image.nii.gz             # 3D Micro-US Volume
│   │   ├── microUS_3D_Subject002_tumor_label.nii.gz       # 3D Micro-US Tumor Label
│   │   ├── Subject002_histopathology_cancer.seg.nrrd      # 3D Histopathology Cancer Labels
│   │   ├── Subject002_histopathology_landmark.seg.nrrd    # 3D Histopathology Landmarks
│   │   ├── Subject002_histopathology_prostate.seg.nrrd    # 3D Histopathology Prostate Label
│   │   ├── Subject002_histopathology_volume.nii.gz        # 3D Histopathology Volume
│   │   ├── Subject002_microUS_landmark.seg.nrrd           # 3D Micro-US Landmarks
│   │   └── Subject002_slice_correspondence.seg.nrrd       # 3D Micro-US Prostate Labels for Corresponding Histopathological Slices
│   ├── Subject006/
│   │   ├── microUS_3D_Subject006_image.nii.gz             # 3D Micro-US Volume
│   │   ├── microUS_3D_Subject006_tumor_label.nii.gz       # 3D Micro-US Tumor Label
│   │   ├── Subject006_histopathology_cancer.seg.nrrd      # 3D Histopathology Cancer Labels
│   │   ├── Subject006_histopathology_landmark.seg.nrrd    # 3D Histopathology Landmarks
│   │   ├── Subject006_histopathology_prostate.seg.nrrd    # 3D Histopathology Prostate Label
│   │   ├── Subject006_histopathology_volume.nii.gz        # 3D Histopathology Volume
│   │   ├── Subject006_microUS_landmark.seg.nrrd           # 3D Micro-US Landmarks
│   │   └── Subject006_slice_correspondence.seg.nrrd       # 3D Micro-US Prostate Labels for Corresponding Histopathological Slices
│   ├── Subject008/
│   └── ...                                                # Similar structure for eighteen subjects
│
├── InteractiveCodes/                                      # Folder for interactive Jupyter notebooks
│   ├── 1_Preparing_the_Data.ipynb                         # Notebook to prepare the dataset for image registration
│   ├── 2_affine_registration_training.ipynb               # Notebook for training deep-learning networks for registration
│   ├── 3_Performance_Evaluation.ipynb                     # Notebook to evaluate and produce registered images and metrics
│   └── utils/                                              # Utility functions required for notebooks
│
└── ScriptedCodes/                                         # Folder for scripted Jupyter notebooks
    ├── 1_Preparing_the_Data.ipynb                         # Notebook to prepare the dataset for image registration
    ├── 2_affine_registration_training.ipynb               # Notebook for training deep-learning networks for registration
    ├── 3_Performance_Evaluation.ipynb                     # Notebook to evaluate and produce registered images and metrics
    └── utils/                                              # Utility functions required for notebooks

```

**Instructions for downloading the dataset:**

Please download the dataset from the provided link and place it in the data directory. Ensure your final data directory matches the structure defined above.

----

<div align=center> <h1>
  Interactive Code
</h1></div>

For those who prefer running scripts from the shell, follow these steps to train the model:


1. **Create an Environment:** Create a new virtual environment using `conda`.
   ```
   conda create --name MicroUS_Hist_Registration python=3.10
   ```
2. **Activate the Enviornment:** Activate the newly created environment.
   ```
   conda activate MicroUS_Hist_Registration
   ```
3. **Install Required Packages:** Install the necessary packages listed in the **[requirements.txt](https://github.com/ImranNust/CIS-UNet-Context-Infused-Swin-UNet-for-Aortic-Segmentation/blob/main/requirements.txt)** file.
   ```
   pip install -r requirements.txt
   ```
4. **Prepare the Dataset:** Prepare the dataset for the training of deep-learning based image registration network.
   ```
   chmod +x ./run_data_preparation.sh
   ./run_data_preparation.sh
   ```
   This code will create two folders `png_images` and `processed_png_data` inside the data directory. The images inside the `processed_png_data` will be used to train the network; therefore, if you like you can delete `png_images` directory.

5. **Train Image Registration Network:**
   - Navigate to the correct directory, where the scripted code is saved:
     ```
     cd ScriptedCodes
     ```
     
    - Now runt the following commands to train both affine and deformable registration networks for six folds:
    
      ```
      chmod +x ./run_training.sh
      ./run_training.sh   
      ```
7. 
## Dependencies

<div align=center>
  
![Python Version](https://img.shields.io/badge/Python-3.10.6-3776AB?logo=python&logoColor=white)
![Open CV Version](https://img.shields.io/badge/OpenCV-4.7.0-5C3EE8?logo=opencv&logoColor=white)
![Numpy Version](https://img.shields.io/badge/Numpy-1.24.4-013243?logo=numpy&logoColor=white)
![PIL Version](https://img.shields.io/badge/PIL-9.2.0-CC3333?logo=python&logoColor=white)
![Matplotlib Version](https://img.shields.io/badge/Matplotlib-3.8.3-11557C?logo=python&logoColor=white)
![Torchvision Version](https://img.shields.io/badge/Torchvision-0.15.1%2Bcu117-EE4C2C?logo=pytorch&logoColor=white)
![Scipy Version](https://img.shields.io/badge/Scipy-1.13.1-8CAAE6?logo=scipy&logoColor=white)
![SimpleITK Version](https://img.shields.io/badge/SimpleITK-2.2.1-8CAAE6?logo=simplitk&logoColor=white)
![CSV Version](https://img.shields.io/badge/CSV-1.0-8CAAE6?logo=csv&logoColor=white)



</div>
