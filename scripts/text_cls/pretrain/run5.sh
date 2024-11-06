dataset="YahooAnswers"
epochs=10

CUDA_VISIBLE_DEVICES=5 python3 text_cls.py --dataset "$dataset" --num-epochs $epochs