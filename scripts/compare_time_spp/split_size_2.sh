parent_dir="saved_cpu"
exp_type="split_size"
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 20000 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 50000 --num_projections 10000 --num_classes 100
python split_cifar3.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 100000 --num_projections 10000 --num_classes 100