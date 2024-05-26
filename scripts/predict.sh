#!/bin/bash

is_tuned="untuned"
strategy="ddp"
upload_user="facebook"
model_type="bart-large-cnn"
text_max_length=1024
precision=32
batch_size=6
epochs="9 10"

for epoch in $epochs
do
    python main.py mode=predict \
        modality=$modality \
        is_tuned=$is_tuned \
        strategy=$strategy \
        upload_user=$upload_user \
        model_type=$model_type \
        text_max_length=$text_max_length \
        precision=$precision \
        batch_size=$batch_size
done
