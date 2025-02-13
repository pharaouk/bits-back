import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


import os
import torch
import numpy as np
import util
import rans
from torch_vae.tvae_binary import BinaryVAE
from torch_vae import tvae_utils
from torchvision import datasets, transforms
from torch.distributions import Bernoulli
import time

rng = np.random.RandomState(0)
np.seterr(all='raise')

randomise_data = False

prior_precision = 8
bernoulli_precision = 12
q_precision = 14

num_images = 200

compress_lengths = []

latent_shape = (40,)
model = BinaryVAE(hidden_dim=100, latent_dim=40)

if randomise_data:
    model.load_state_dict(
        torch.load('/content/drive/My Drive/Colab Notebooks/bits-back/torch_vae/saved_params/torch_binary_vae_params_random'))
else:
    model.load_state_dict(
        torch.load('/content/drive/My Drive/Colab Notebooks/bits-back/torch_vae/saved_params/torch_binary_vae_params'))

rec_net = tvae_utils.torch_fun_to_numpy_fun(model.encode)
gen_net = tvae_utils.torch_fun_to_numpy_fun(model.decode)

obs_append = tvae_utils.bernoulli_obs_append(bernoulli_precision)
obs_pop = tvae_utils.bernoulli_obs_pop(bernoulli_precision)

vae_append = util.vae_append(latent_shape, gen_net, rec_net, obs_append,
                             prior_precision, q_precision)
vae_pop = util.vae_pop(latent_shape, gen_net, rec_net, obs_pop,
                       prior_precision, q_precision)

if __name__ == '__main__':
    # load some mnist images
    mnist = datasets.MNIST('data/mnist', train=False, download=True,
                           transform=transforms.Compose(
                               [transforms.ToTensor()]))
    images = mnist.test_data[:num_images]

    if randomise_data:
        images = Bernoulli(images.float() / 255.).sample()
    else:
        images = torch.round(images.float() / 255.)

    images = [image.view(-1) for image in images]

    # randomly generate some 'other' bits
    other_bits = rng.randint(low=1 << 16, high=1 << 31, size=20, dtype=np.uint32)
    state = rans.unflatten(other_bits)

    print_interval = 10
    encode_start_time = time.time()
    for i, image in enumerate(images):
        state = vae_append(state, image)

        if not i % print_interval:
            print('Encoded {}'.format(i))

        compressed_length = 32*(len(rans.flatten(state)) - len(other_bits)) / (i+1)
        compress_lengths.append(compressed_length)

    print('\nAll encoded in {:.2f}s'.format(time.time() - encode_start_time))
    compressed_message = rans.flatten(state)

    compressed_bits = 32 * (len(compressed_message) - len(other_bits))
    print("Used " + str(compressed_bits) +
          " bits.")
    print('This is {:.2f} bits per pixel'.format(compressed_bits / (num_images*784.)))

    if not os.path.exists('results'):
        os.mkdir('results')
    np.savetxt('results/compressed_lengths_bin', np.array(compress_lengths))

    state = rans.unflatten(compressed_message)
    decode_start_time = time.time()

    for n in range(len(images)):
        state, image_ = vae_pop(state)
        original_image = images[len(images)-n-1].numpy()
        assert all(original_image == np.array(image_))

        if not n % print_interval:
            print('Decoded {}'.format(n))

    print('\nAll decoded in {:.2f}s'.format(time.time() - decode_start_time))

    recovered_bits = rans.flatten(state)
    assert all(other_bits == recovered_bits)
