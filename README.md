# Novel Molecular Representations using Neumann-Cayley Orthogonal Gated Recurrent Unit
+ Full Code is going to be available soon

Implementation of the Paper "Novel Molecular Representations using Neumann-Cayley Orthogonal Gated Recurrent Unit" by 

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
The google_drive_data.zip file can also be downloaded manualy under https://drive.google.com/drive/folders/1GwcGNWcl8IbY0TnFfw9U02wWthRJncdf

## Train AutoEncoder from scratch
Run the script run.sh :
```bash
bash run.sh
```
