# Setup-NVIDIA-GPU-for-Deep-Learning

## Step 1: NVIDIA Video Driver

You should install the latest version of your GPUs driver. You can download drivers here:
 - [NVIDIA GPU Drive Download](https://www.nvidia.com/Download/index.aspx)

## Step 2: Visual Studio C++

You will need Visual Studio, with C++ installed. By default, C++ is not installed with Visual Studio, so make sure you select all of the C++ options.
 - [Visual Studio Community Edition](https://visualstudio.microsoft.com/vs/community/)

## Step 3: Anaconda/Miniconda

You will need anaconda to install all deep learning packages
 - [Download Anaconda](https://www.anaconda.com/download/success)

## Step 4: CUDA Toolkit

 - [Download CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive)

## Step 5: cuDNN

 - [Download cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)


## Step 6: Install PyTorch 

 - [Install PyTorch](https://pytorch.org/get-started/locally/)




## Finally run the following script to test your GPU

```python
import torch

print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())
```
```
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
```
```
if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(device)
    properties = torch.cuda.get_device_properties(device)
    sm_count = properties.multi_processor_count
    cores_per_sm = {
        'Turing': 64,  # RTX 20xx, GTX 16xx series
        'Ampere': 128, # RTX 30xx series
        'Pascal': 128, # GTX 10xx series
    }

    architecture = properties.name.split()[0]  
    cores = cores_per_sm.get(architecture, 64)  
    total_cores = sm_count * cores

    print(f"GPU Name: {gpu_name}")
    print(f"Architecture: {architecture}")
    print(f"Number of CUDA Cores: {total_cores}")
else:
    print("CUDA is not available on this device.")
```
