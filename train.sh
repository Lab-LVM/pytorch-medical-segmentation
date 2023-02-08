#### TransUnet
python -W ignore train_ACDC.py -m TransUnet --cuda 1 --use-wandb
python -W ignore train_polyp.py -m TransUnet --cuda 1 --use-wandb
python -W ignore train_synapse.py -m TransUnet --cuda 1 --use-wandb

#### TransCASCADE
python -W ignore train_ACDC.py -m TransCASCADE --cuda 1 --use-wandb
python -W ignore train_polyp.py -m TransCASCADE --cuda 1 --use-wandb
python -W ignore train_synapse.py -m TransCASCADE --cuda 1 --use-wandb

#### TransCASCADE-pt (load imagenet21k pretrained wieghts to Resnet50 and ViT-base)
python -W ignore train_ACDC.py -m TransCASCADE-pt --cuda 1 --use-wandb
python -W ignore train_polyp.py -m TransCASCADE-pt --cuda 1 --use-wandb
python -W ignore train_synapse.py -m TransCASCADE-pt --cuda 1 --use-wandb

#### PVT-CASCADE
python -W ignore train_ACDC.py -m PVT-CASCADE --cuda 1 --use-wandb
python -W ignore train_polyp.py -m PVT-CASCADE --cuda 1 --use-wandb
python -W ignore train_synapse.py -m PVT-CASCADE --cuda 1 --use-wandb

#### PVT-CASCADE-pt (load imagenet21k pretrained wieghts to PVT backbone)
python -W ignore train_ACDC.py -m PVT-CASCADE-pt --cuda 1 --use-wandb
python -W ignore train_polyp.py -m PVT-CASCADE-pt --cuda 1 --use-wandb
python -W ignore train_synapse.py -m PVT-CASCADE-pt --cuda 1 --use-wandb

################################################################################
# if you want to apply multi-way to each model,
# add "multiway" in front of model name
# and also add option of multiway at the end of this model name.
#   e.g.  -m multiway-TranUnet-d192-d0-p3-concat
################################################################################
python -W ignore train_synapse.py -m multiway-TransUnet-d192-d0-p3-concat --cuda 1 --use-wandb
python -W ignore train_synapse.py -m multiway-TransUnet-d192-d0-p3-add --cuda 1 --use-wandb