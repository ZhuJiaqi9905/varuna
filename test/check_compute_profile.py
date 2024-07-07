import pickle

# with open('/mnt/gpu-91/varuna/profiles_bak/compute-profile-10', 'rb') as file:
#     data = pickle.load(file)
#     print(data)
#     print(len(data))

for i in range(0, 8):
    with open(f"/mnt/gpu-91/varuna/profile_rank_{i}/_tmp_inp_shapes",'rb') as f:
        input_shapes = pickle.load(f)
        input_shapes_keys = list(input_shapes.keys())
        input_shapes_res = [input_shapes[k][0] for k in input_shapes_keys]
        print(input_shapes_res, len(input_shapes_res) + 1)