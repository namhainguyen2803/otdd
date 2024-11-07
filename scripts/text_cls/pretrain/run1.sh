dataset="AG_NEWS"
epochs=10

CUDA_VISIBLE_DEVICES=4 python3 text_cls.py --dataset "$dataset" --num-epochs $epochs