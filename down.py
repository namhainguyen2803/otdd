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

# x = torch.randn(1, 1, 28, 28).to('cpu')

# U_list = generate_unit_convolution_projections(image_size=28, num_channels=1, num_projection=100, device='cpu')

# for U in U_list:
#     x = U(x)

# print(x.shape)

NUM_EXAMPLES = 3
NUM_PROJECTION = 4
NUM_MOMENTS = 2

proj_matrix_dataset = list()
for i in range(2):
    X_projection = torch.randint(1, 4, size=(NUM_EXAMPLES, NUM_PROJECTION))
    k = torch.randint(1, 4, size=(NUM_PROJECTION, NUM_MOMENTS))
    print(X_projection.t())
    print(k)
    moment_X_projection = torch.pow(input=X_projection.unsqueeze(1), exponent=k.permute(1, 0))
    print(moment_X_projection)
    moment_X_projection = moment_X_projection.permute(2, 0, 1) 
    print(moment_X_projection)
    avg_moment_X_projection = torch.sum(moment_X_projection, dim=1)
    print(avg_moment_X_projection)

    X_projection = torch.permute(X_projection, dims=(1, 0)) # shape == (num_projection, num_examples)
    h = torch.cat((X_projection.unsqueeze(-1),
                    avg_moment_X_projection.unsqueeze(1).expand(NUM_PROJECTION, NUM_EXAMPLES, NUM_MOMENTS)), 
                    dim=2) 
    # shape == (num_projection, num_examples, num_moments+1)
    print(h.shape)
    print(h)
    # print(h.permute(1,0,2))
    proj_matrix_dataset.append(h)
    print("---------")

proj_matrix_dataset = torch.cat(proj_matrix_dataset, dim=1) 
print(proj_matrix_dataset)

projection_matrix_2 = torch.randn(NUM_PROJECTION, NUM_MOMENTS+1)
proj_matrix_dataset = proj_matrix_dataset.type(torch.FloatTensor)
proj_proj_matrix_dataset = torch.matmul(proj_matrix_dataset, projection_matrix_2.unsqueeze(-1)).squeeze(-1) # shape == (num_projection, total_examples)

print("==========")
print(proj_proj_matrix_dataset)
print("=========")

for i in range(NUM_PROJECTION):
    print(torch.matmul(proj_matrix_dataset[i, :, :], projection_matrix_2[i, :]))