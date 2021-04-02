import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)



#import os
import re
import torch
import numpy as np
import util
import rans
from torch_vae.tvae_beta_binomial import BetaBinomialVAE
from torch_vae import tvae_utils
from torchvision import datasets, transforms
import time


def string_to_array(my_string):
    my_string = my_string.lower()
    my_string = re.sub('[^acgt]', 'z', my_string)
    my_array = np.array(list(my_string))
    return my_array

# create a label encoder with 'acgtn' alphabet
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['a','c','g','t','z']))

# function to encode a DNA sequence string as an ordinal vector
# returns a numpy vector with a=0.25, c=0.50, g=0.75, t=1.00, n=0.00
def ordinal_encoder(my_array):
    integer_encoded = label_encoder.transform(my_array)
    float_encoded = integer_encoded.astype(float)
    float_encoded[float_encoded == 0] = 25 # A
    float_encoded[float_encoded == 1] = 100 # C
    float_encoded[float_encoded == 2] = 150 # G
    float_encoded[float_encoded == 3] = 250 # T
    float_encoded[float_encoded == 4] = 0.00 # anything else, z
    return float_encoded


with open('/content/drive/My Drive/Colab Notebooks/bits-back/torch_vae/data/dna/dnafrgmnt00.txt', 'r') as file:
    data = file.read().replace('/n', '')
    data = data.replace(' ', '')

split_strings = []
n  = 784
for index in range(0, len(data), n):
    split_strings.append(data[index : index + n])

dna_arrays = []
for i in range(len(split_strings)):
  y = ordinal_encoder(string_to_array(split_strings[i]))
  dna_arrays.append(y)

dna_arrays = np.array(dna_arrays)
print(dna_arrays.shape)

l = np.vstack( dna_arrays )
print(l.shape)
x = torch.from_numpy(l)
print(x.shape)








rng = np.random.RandomState(0)
np.seterr(over='raise')

prior_precision = 8
obs_precision = 14
q_precision = 14

num_images = 100

compress_lengths = []

latent_dim = 50
latent_shape = (1, latent_dim)
model = BetaBinomialVAE(hidden_dim=200, latent_dim=latent_dim)
model.load_state_dict(
    torch.load('/content/drive/My Drive/Colab Notebooks/bits-back/torch_vae/saved_params/torch_vae_beta_binomial_params',
               map_location=lambda storage, location: storage))
model.eval()

rec_net = tvae_utils.torch_fun_to_numpy_fun(model.encode)
gen_net = tvae_utils.torch_fun_to_numpy_fun(model.decode)

obs_append = tvae_utils.beta_binomial_obs_append(255, obs_precision)
obs_pop = tvae_utils.beta_binomial_obs_pop(255, obs_precision)

vae_append = util.vae_append(latent_shape, gen_net, rec_net, obs_append,
                             prior_precision, q_precision)
vae_pop = util.vae_pop(latent_shape, gen_net, rec_net, obs_pop,
                       prior_precision, q_precision)




images = x[:num_images]



images = [image.float().view(1, -1) for image in images]
print(images[0])
print(images[0].shape)
# randomly generate some 'other' bits
other_bits = rng.randint(low=1 << 16, high=1 << 31, size=50, dtype=np.uint32)
state = rans.unflatten(other_bits)


print_interval = 10
encode_start_time = time.time()
for i, image in enumerate(images):
    state = vae_append(state, image)

    if not i % print_interval:
        print('Encoded {}'.format(i))

    compressed_length = 32 * (len(rans.flatten(state)) - len(other_bits)) / (i+1)
    compress_lengths.append(compressed_length)

print('\nAll encoded in {:.2f}s'.format(time.time() - encode_start_time))
compressed_message = rans.flatten(state)

compressed_bits = 32 * (len(compressed_message) - len(other_bits))
print("Used " + str(compressed_bits) + " bits.")
print('This is {:.2f} bits per pixel'.format(compressed_bits
                                             / (num_images * 784)))

if not os.path.exists('results'):
    os.mkdir('results')
np.savetxt('compressed_lengths_cts', np.array(compress_lengths))
compressed_message = np.loadtxt('compressed_message')

state = rans.unflatten(compressed_message)
decode_start_time = time.time()
for n in range(len(images)):    
    state, image_ = vae_pop(state)
    original_image = images[len(images)-n-1].numpy()
    
    np.testing.assert_allclose(original_image, image_)
    np.allclose(original_image, image_)
    print(np.allclose(original_image, image_))

    if not n % print_interval:
        print('Decoded {}'.format(n))

print('\nAll decoded in {:.2f}s'.format(time.time() - decode_start_time))

recovered_bits = rans.flatten(state)

print(recovered_bits)
assert all(other_bits == recovered_bits)
