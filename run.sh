#!/usr/bin/env bash


##################################### MobileNet_V2 ################################################

#1- FCN8s MobileNet_V2 Train
python3 main.py --load_config=fcn8s_mobilenetV2_train.yaml train Train FCN8sMobileNet_V2

