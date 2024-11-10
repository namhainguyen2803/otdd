parent_dir="saved_cifar10"
exp_type="split_size"
python split_cifar10.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 1000 --num_projections 10000 --num_classes 10
python split_cifar10.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 3000 --num_projections 10000 --num_classes 10
python split_cifar10.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 5000 --num_projections 10000 --num_classes 10
python split_cifar10.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 7000 --num_projections 10000 --num_classes 10
python split_cifar10.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 9000 --num_projections 10000 --num_classes 10
python split_cifar10.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 11000 --num_projections 10000 --num_classes 10
python split_cifar10.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 13000 --num_projections 10000 --num_classes 10
python split_cifar10.py --parent_dir "$parent_dir" --exp_type "$exp_type" --num_splits 2 --split_size 15000 --num_projections 10000 --num_classes 10