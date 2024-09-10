dataset="YelpReviewFull"
epochs=10

CUDA_VISIBLE_DEVICES=6 python3 text_cls.py --dataset "$dataset" --num-epochs $epochs