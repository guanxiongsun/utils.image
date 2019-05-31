import numpy as np
import skimage.io as skio
import os

imgs = {
    'lena_png': 'imgs/lena.png',
    'lena_color': 'imgs/lena_color.gif',
    'lena_gray': 'imgs/lena_gray.gif',
    'bear_jpg': 'imgs/COCO_val2014_000000000285.jpg'
}


def load(img_path, mode='RGB'):
    """
    Load an image
    :param img_path: path of the image file
    :param mode: 'RGB' or 'GRAY'
    :return: np.ndarray [H, W, C], channel sequence is [RGB]
             dtype=uint8
    """
    if not os.path.isfile(img_path):
        raise ValueError("{} not exist.".format(img_path))
    if mode == 'RGB':
        return skio.imread(img_path)
    elif mode == 'L' or mode == 'GRAY':
        return skio.imread(img_path, as_gray=True)
    else:
        raise ValueError("Mode should be 'RGB' or 'GRAY")


def load_pil(img_path):
    """
    Load an image and return in PIL.Image format
    :param img_path:
    :return:
    """
    import PIL.Image
    if not os.path.isfile(img_path):
        raise ValueError("{} not exist.".format(img_path))
    return PIL.Image.open(img_path)


def load_cv(img_path):
    """
    Load an image and return in cv_mat format
    :param img_path:
    :return:
    """
    if not os.path.isfile(img_path):
        raise ValueError("{} not exist.".format(img_path))
    import cv2
    _img_cv = cv2.imread(img_path)

    # Some formats are not supported in opencv and will return None
    if _img_cv is None:
        _img_cv = np2cv(load(img_path))

    return _img_cv


def save(np_img, target_path):
    """
    Save np img to target path
    :param np_img:
    :param target_path:
    :return:
    """
    skio.imsave(target_path, np_img)


def save_pil(pil_img, target_path):
    _np_img = pil2np(pil_img)
    save(_np_img, target_path)


def save_cv(cv_img, target_path):
    _np_img = cv2np(cv_img)
    save(_np_img, target_path)


def pil2np(pil_img):
    """
    Convert a PIL.Image to numpy img.
    Note:
        a PIL.Image object's size is [W, H], so the index is (x, y) opposite to common imgs.
    :param pil_img: PIL.Image object
    :return:
    """
    import PIL.Image
    if isinstance(pil_img, PIL.Image.Image):
        return np.asarray(pil_img)
    else:
        raise TypeError("Input must be a PIL.Image object")


def img_255(np_img):
    """
    Change the pixel value from [0,1] float32 to [0,255] uint8
    :param np_img: float32
    :return: np img uint8
    """
    if np_img.dtype == np.uint8:
        return np_img
    elif np_img.dtype == np.float32:
        return np.array(np_img*255, dtype=np.uint8)
    else:
        raise TypeError("Unsupported dtype of np img")


def img_1(np_img):
    """
    Change the pixel value from [0,255] unit8 to [0,1] uint8
    :param np_img: uint8
    :return: np img float32
    """
    if np_img.dtype == np.float32:
        return np_img
    elif np_img.dtype == np.uint8:
        return np.array(np_img/255.0, dtype=np.float32)
    else:
        raise TypeError("Unsupported dtype of np img")


def np2pil(np_img):
    """
    Convert a np img to pil img
    :param np_img:
    :return: pil img
    """
    import PIL.Image
    if np_img.dtype == np.uint8:
        return PIL.Image.fromarray(np_img)
    elif np_img.dtype == np.float32:
        return PIL.Image.fromarray(img_255(np_img))
    else:
        raise TypeError("Unsupported dtype of np img")


def reverse_channel(np_img, dim=2):
    """
    Reverse the channel sequence.
    :param np_img: np img
    :param dim: target dimension to change the sequence
    :return: np img
    """
    if dim == 0:
        return np_img[[2,1,0], :,:]
    elif dim == 1:
        return np_img[:, [2,1,0], :]
    elif dim == 2:
        return np_img[:,:,[2,1,0]]
    else:
        raise ValueError("dim should be in [0,1,2]")


# def bgr2rgb(np_img, dim=2):
#     """
#     Channel sequence [BGR] to [RGB]
#     :param np_img: [BGR]
#     :param dim: target dimension to change sequence
#     :return: np_img [RGB]
#     """
#     return reverse_channel(np_img, dim)


def cv2np(cv_img):
    """
    Convert a cv_mat image to numpy img.
    Note:
        a cv_mat image is already numpy ndarray, except the channel sequence is [BGR] in opencv.
        so, this function is to reverse the sequence on a specific dimension.
    :param cv_img: cv_mat img
    :return: numpy img
    """
    if isinstance(cv_img, np.ndarray):
        if cv_img.ndim == 3:
            # color image
            if cv_img.shape[2] == 3:
                return reverse_channel(cv_img, dim=2)
            else:
                raise ValueError("The number of dim 2 in cv_img is not 3.")
        elif cv_img.ndim == 2:
            # gray image
            return cv_img
    else:
        raise TypeError("Input must be a cv_mat image.")


def np2cv(np_img, dim=2):
    """
    Convert a np img back to cv_mat img
    :param np_img: [RGB]
    :param dim: target channel dimension
    :return: [BGR]
    """
    if np_img.ndim == 3:
        # color image
        if np_img.shape[dim] is not 3:
            raise ValueError("Channel number of target dim is not 3.")
        return reverse_channel(np_img, dim)
    elif np_img.ndim == 2:
        # gray image
        return np_img


def convert_color(np_img, mode=None, alpha=None):
    import skimage.color as skcolor
    if mode == 'rgb2gray':
        return skcolor.rgb2gray(np_img).astype(np.float32)
    elif mode == 'gray2rgb':
        return skcolor.gray2rgb(np_img, alpha).astype(np.float32)
    else:
        raise NotImplementedError


def rgb2gray(np_img):
    """
    Convert a rgb np img to gray
    :param np_img: [RGB]
    :return: np img gray
    """
    return convert_color(np_img, mode='rgb2gray')


def gray2rgb(np_img, alpha=None):
    """
    Convert a gray np img to rgb
    :param np_img: gray
    :param alpha: alpha
    :return: np img [RGB]
    """
    return convert_color(np_img, mode='gray2rgb', alpha=alpha)


def cv_show(cv_img, title=None, delay=3000):
    import cv2
    cv2.imshow(title, cv_img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()


def pil_show(pil_img):
    pil_img.show()


def plt_show(np_img):
    skio.imshow(np_img)
    skio.show()


def show(np_img, backend='plt', title=None, delay=3000):
    if backend == 'plt':
        plt_show(np_img)
    elif backend == 'pil':
        pil_show(np2pil(np_img))
    elif backend == 'cv2':
        cv_show(np2cv(np_img), title, delay)
    else:
        raise ValueError("Backend should be in ['plt', 'pil', 'cv2']")


if __name__ == '__main__':

    # PNG format color image, all fine
    # lena = imgs['lena_png']

    # GIF format gray image, PIL image is different.
    # lena = imgs['lena_gray']

    # JPEG format color image
    lena = imgs['bear_jpg']

    try:
        lena_cv = load_cv('unexisted/path.jpg')
    except:
        lena_cv = load_cv(lena)

    try:
        lena_pil = load_pil('unxfdfs/sdfsf/sdfsdf/g.jjj')
    except:
        lena_pil = load_pil(lena)

    lena_np = load(lena)

    lena_cv_np = cv2np(lena_cv)
    lena_pil_np = pil2np(lena_pil)

    if np.all(lena_cv_np == lena_pil_np):
        print("lena_cv_np is equal to lena_pil_np!")

    cv_show(lena_cv)
    cv_show(lena_cv_np)
    pil_show(lena_pil)
    show(lena_cv_np)

    show(lena_pil_np, backend='cv2')
    show(lena_pil_np, backend='pil')
    show(lena_pil_np, backend='plt')
    show(img_1(lena_cv_np), backend='cv2', title='cv2', delay=0)
    show(img_1(img_1(img_255(img_255(img_1(lena_cv_np))))), backend='pil')
    show(img_1(lena_cv_np), backend='plt')

    lena_rgb2gray = rgb2gray(lena_np)
    lena_gray2rgb = gray2rgb(lena_rgb2gray)

    show(lena_rgb2gray)
    show(lena_gray2rgb)

    print("done")