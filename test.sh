################################################################################
# if you want to test model performance,
# find a experiment name in ./model_pth/{dataset name}/
# and use the experiment name as value of --exp option
#   e.g.  --exp TransUnet_ACDC224_v1
################################################################################
python -W ignore test_synapse.py --exp TransUnet_ACDC224_v1 --cuda 3