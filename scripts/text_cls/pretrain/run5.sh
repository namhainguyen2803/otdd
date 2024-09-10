dataset="YahooAnswers"
epochs=10

CUDA_VISIBLE_DEVICES=7 python3 text_cls.py --dataset "$dataset" --num-epochs $epochs