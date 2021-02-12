import numpy as np
import re
import sys
import struct


def read_image(path):
    if path.endswith('.bin'):
        return read_bin(path)
    elif path.endswith('.pfm'):
        return read_pfm(path)
    else:
        raise Exception('Invalid input format; only pfm and bin are supported')


def save_image(path, image):
    if path.endswith('.bin'):
        save_bin(path, image)
    elif path.endswith('.pfm'):
        save_pfm(path, image)
    else:
        raise Exception('Invalid input format; only pfm and bin are supported')


def read_bin(path):
    with open(path, 'rb') as fid:
        width, height, channels = np.genfromtxt(fid, delimiter='&', max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b'&':
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        data = np.fromfile(fid, np.float32)
    data = data.reshape((width, height, channels), order='F')
    data = np.transpose(data, (1, 0, 2))
    return data, 1


def save_bin(path, image):
    if image.dtype != np.float32:
        raise Exception('Image data type must be float32.')

    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        height, width = image.shape
        channels = 1
    elif len(image.shape) == 3 and image.shape[2] == 3:
        height, width, channels = image.shape
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    with open(path, 'w') as fid:
        fid.write(str(width) + '&' + str(height) + '&' + str(channels) + '&')

    with open(path, 'ab') as fid:
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            image_trans = np.transpose(image, (1, 0))
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image_trans = np.transpose(image, (1, 0, 2))
        else:
            raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')
        data_1d = image_trans.reshape(-1, order='F')
        data_list = data_1d.tolist()
        endian_character = '<'
        format_char_sequence = ''.join(['f'] * len(data_list))
        byte_data = struct.pack(endian_character + format_char_sequence, *data_list)
        fid.write(byte_data)


def read_pfm(filename):
    # rb: binary file and read only
    file = open(filename, 'rb')

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':  # depth is Pf
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))  # re is used for matching
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width, 1)
    # depth: H*W
    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")

    image = np.flipud(image)
    # print(image.shape)

    if image.dtype.name != 'float32':
        raise Exception('Image data type must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()
