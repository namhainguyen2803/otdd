# import sentence_transformers as st

# dpath = "data/ag_news_csv"
# reader = st.readers.LabelSentenceReader(dpath)

# s = reader.get_examples('test.tsv')

# for i in range(len(s)):
#     if s[i].label == 0:
#         print(s[i].texts, s[i].label)


import torch
from otdd.pytorch.utils import generate_unit_convolution_projections, generate_uniform_unit_sphere_projections

# num_moments = 3
# num_examples = 4
# num_projection = 2
# moments = torch.randint(1, num_moments + 1, (num_projection, num_moments))

# x = torch.randint(1, 5, (num_examples, num_projection))
# x_pow = torch.pow(x.unsqueeze(1), moments.permute(1, 0)).permute(2, 0, 1)


# print(moments)

# print(x.permute(1, 0))

# print(x_pow, x_pow.shape)

x = torch.randn(1, 1, 28, 28).to('cpu')

U_list = generate_unit_convolution_projections(image_size=28, num_channels=1, num_projection=100, device='cpu')

for U in U_list:
    x = U(x)

print(x.shape)