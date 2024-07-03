import pickle

with open('/mnt/gpu-91/varuna/profiles_bak/compute-profile-0', 'rb') as file:
    print(pickle.load(file))