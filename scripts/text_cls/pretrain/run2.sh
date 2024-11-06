dataset="DBpedia"
epochs=10

CUDA_VISIBLE_DEVICES=1 python3 text_cls.py --dataset "$dataset" --num-epochs $epochs