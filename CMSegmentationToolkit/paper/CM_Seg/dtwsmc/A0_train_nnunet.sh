#!/bin/bash
nnUNetv2_plan_and_preprocess -d 821 -c 3d_fullres
nnUNetv2_plan_and_preprocess -d 822 -c 3d_fullres
nnUNetv2_train 821 3d_fullres 0 1 2 3 4
nnUNetv2_train 822 3d_fullres 0 1 2 3 4