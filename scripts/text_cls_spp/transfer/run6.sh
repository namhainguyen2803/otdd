dataset="AmazonReviewPolarity"
epochs=2

CUDA_VISIBLE_DEVICES=5 python3 text_cls_transfer.py --dataset "$dataset" --num-epochs $epochs