import torch


data_dir = "saved_text_dataset"
dataset_name = "AG_NEWS"

path = f"{data_dir}/{dataset_name}.pt"

embed, label = torch.load(path)
print(embed)
print(embed.shape, label.shape)

print(embed.min(), embed.max())