import logging
import os.path

import requests
import numpy as np
logger = logging.getLogger(__name__)

def convert_image_by_pixformat_normalize(src_image):
    # Convert to NCHW format
    src_image = src_image.transpose((2, 0, 1))
    # Normalize to -1 to 1
    src_image = (src_image.astype(np.float) / 255) * 2.0 - 1.0
    return src_image

def maybe_download(url, dest):
    """Download the url to dest if necessary, optionally checking file
    integrity.
    """
    if not os.path.exists(dest):
        logger.info('Downloading %s to %s', url, dest)
        download(url, dest)

def download(url, dest):
    """Download the url to dest, overwriting dest if it already exists."""
    response = requests.get(url, stream=True)
    with open(dest, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

# TODO: Poor naming: Change this.
# TODO: WTF does get_transform2 mean.
def get_transform2(dataset_name, net_transform, downscale):
    "Returns image and label transform to downscale, crop and prepare for net."
    orig_size = get_orig_size(dataset_name)
    transform = []
    target_transform = []
    if downscale is not None:
        transform.append(transforms.Resize(orig_size // downscale))
        target_transform.append(
                transforms.Resize(orig_size // downscale,
                    interpolation=Image.NEAREST))

    transform.extend([transforms.Resize(orig_size), net_transform]) 
    target_transform.extend([transforms.Resize(orig_size, interpolation=Image.NEAREST),
        to_tensor_raw]) 

    transform = transforms.Compose(transform)
    target_transform = transforms.Compose(target_transform)

    return transform, target_transform

# TODO: Poor Naming. Change this.
def get_transform(params, image_size, num_channels):
    # Transforms for PIL Images: Gray <-> RGB
    Gray2RGB = transforms.Lambda(lambda x: x.convert('RGB'))
    RGB2Gray = transforms.Lambda(lambda x: x.convert('L'))

    transform = []
    # Does size request match original size?
    if not image_size == params.image_size:
        transform.append(transforms.Resize(image_size))
   
    # Does number of channels requested match original?
    if not num_channels == params.num_channels:
        if num_channels == 1:
            transform.append(RGB2Gray)
        elif num_channels == 3:
            transform.append(Gray2RGB)
        else:
            print('NumChannels should be 1 or 3', num_channels)
            raise Exception

    transform += [transforms.ToTensor(), 
            transforms.Normalize((params.mean,), (params.std,))]

    return transforms.Compose(transform)


def get_target_transform(params):
    transform = params.target_transform
    t_uniform = transforms.Lambda(lambda x: x[:,0] 
            if isinstance(x, (list, np.ndarray)) and len(x) == 2 else x)
    if transform is None:
        return t_uniform
    else:
        return transforms.Compose([transform, t_uniform])
