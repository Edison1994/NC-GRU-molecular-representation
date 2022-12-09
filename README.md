# Novel Molecular Representations using Neumann-Cayley Orthogonal Gated Recurrent Unit
+ Full Code is going to be available soon

Implementation of the Paper "Novel Molecular Representations using Neumann-Cayley Orthogonal Gated Recurrent Unit" by Edison Mucllari, Vasily Zadorozhnyy, Qiang Ye and Duc Nguyen

## Installing

### Conda
Create a new ncgru_fp enviorment:
```bash
git clone https://github.com/Edison1994/NC-GRU-molecular-representation.git
conda env create -f env.yml
source activate ncgru_fp
```

Install tensorflow with GPU support:
```bash
pip install tensorflow-gpu==1.10.0
```

Install additional zmq package:
```bash
pip install zmq
```

## Getting Data (from Google Drive)
```bash

```
The google_drive_data.zip file can also be downloaded manualy under https://drive.google.com/file/d/1pXBp7Jvf9iS6reQbv5MxJ5z4f0lpwWaD/view?usp=share_link

After unzipping the folder, chembl_28 corresponds to the AutoEncoder train/test data and prediction_data is the folder containing the data for the inference model 

## Train AutoEncoder from scratch
Run the script run.sh :
```bash
bash run.sh
```
