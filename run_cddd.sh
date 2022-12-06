python run_cddd.py \
--input prediction_data/LD50_train.smi \
--model_dir AutoEncoders_pretrained_models/160_320_640_glorot_uniform \
--output mtdnn_for_ld50_igc50/_csv_/LD50_NCGRU_FP_train2.csv \
--smiles_header smiles \
--use_gpu

# python run_cddd.py \
# --input prediction_data/<prediction dataset, see below> \
# --model_dir AutoEncoders_pretrained_models/<your model or pretrained model from below> \
# --output mtdnn4/_csv_/<your filename of descriptors>.csv \
# --smiles_header smiles \
# --use_gpu

### Choose one of the prediction datasets ###
#   LD50:             LD50_all.smi            LD50_train.smi                                          LD50_test.smi
#   IGC50:            IGC50_all.smi           IGC50_train.smi                                         IGC50_test.smi
#   LC50:             LC50_all.smi            LC50_train.smi                                          LC50_test.smi
#   LC50DM:           LC50DM_all.smi          LC50DM_train.smi                                        LC50DM_test.smi
#   logP:             logP_all.smi            logP_train.smi                                          logP_test.smi
#   FreeSolv:         FreeSolv_all.smi        FreeSolv_train.smi          FreeSolv_valid.smi          FreeSolv_test.smi
#   Lipophilicity:    Lipophilicity_all.smi   Lipophilicity_train.smi     Lipophilicity_valid.smi     Lipophilicity_test.smi

### Chooose one of the pretrained models ###
#   --model_dir AutoEncoders_pretrained_models/160_320_640_glorot_uniform \
#   --model_dir AutoEncoders_pretrained_models/160_320_640_he_normal \
#   --model_dir AutoEncoders_pretrained_models/160_320_glorot_uniform \
#   --model_dir AutoEncoders_pretrained_models/160_320_he_normal \

### EXAMPLE: To extract NC-GRU FingerPrints for training LD50 dataset using 160_320_640_glorot_uniform model you would run: ###
# python run_cddd.py \
# --input prediction_data/LD50_train.smi \
# --model_dir AutoEncoders_pretrained_models/160_320_640_glorot_uniform \
# --output mtdnn_for_ld50_igc50/_csv_/LD50_NCGRU_FP_train.csv \
# --smiles_header smiles \
# --use_gpu