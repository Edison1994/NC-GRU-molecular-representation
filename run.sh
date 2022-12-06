python train.py \
--input_sequence_key "canonical_smiles" \
--cell_size 160 320 \
--nono 40 80 \
--train_file chembl_28_train.tfrecords \
--val_file chembl_28_test.tfrecords \
--one_hot_embedding True \
--emb_size 512 \
--device 0 \
--model "NoisyGRUSeq2SeqWithFeatures" 
