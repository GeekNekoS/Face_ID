from PIL import Image


def resize_image(image: Image, new_size: tuple[int, int]):
    """
    Resize image.
    :param image: PIL Image, image
    :param new_size: tuple, new image size
    :return: PIL Image, image with new size
    """
    image = image.thumbnail(new_size)
    return image
