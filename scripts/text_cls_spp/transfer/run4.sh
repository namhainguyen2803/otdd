dataset="YelpReviewFull"
epochs=2

CUDA_VISIBLE_DEVICES=3 python3 text_cls_transfer.py --dataset "$dataset" --num-epochs $epochs