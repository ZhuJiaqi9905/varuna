import pickle

with open('/mnt/gpu-91/varuna/profiles_bak/compute-profile-10', 'rb') as file:
    data = pickle.load(file)
    print(data)
    print(len(data))