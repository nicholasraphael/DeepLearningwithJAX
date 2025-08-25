#%%
# einops



# Matrix mul dominate the compute 2 * m * n * p FLOPs
# FLOP/s depends on hardware and data types

#  num_of_forward_flops = (2 * params_1) + (2 * params_2) + ...
# num_of_backward_flops = 4 * params