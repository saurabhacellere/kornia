import torch
import torch.nn as nn


class RgbToYcbcr(torch.nn.Module):
    r"""Convert image from RGB to YCbCr.
    The image data is assumed to be in the range of (0, 1).

    args:
        image (torch.Tensor): RGB iRGB image to be converted to YCbCr.

    returns:
        torch.Tensor: YCbCr version of the image.

    shape:
        - image: :math:`(*, 3, H, W)`
        - output: :math:`(*, 3, H, W)`

    """

    def __init__(self) -> None:
        super(RgbToYcbcr, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return rgb_to_ycbcr(image)


def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a RGB image to YCbCr.

    The image data is assumed to be in the range of (0, 1).

    See :class:`~kornia.color.RgbToYcbcr` for details.

    Args:
        input (torch.Tensor): RGB Image to be converted to YCbCr. The image has to
        be in the shape of (*, 3, H, W).

    Returns:
        torch.Tensor: YCbCr version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W).Got {}"
                         .format(image.shape))

    # unpack the rgb values
    r, g, b = torch.chunk(image, dim=-3, chunks=3)

    # apply weights
    # reference: https://codeday.me/es/qa/20190322/352983.html
    delta: float = 0.5

    y: torch.Tensor  =  0.299000 * r + 0.587000 * g + 0.114000 * b
    cb: torch.Tensor = -0.168736 * r - 0.331364 * g + 0.500000 * b + delta
    cr: torch.Tensor =  0.500000 * r - 0.418688 * g - 0.081312 * b + delta

    return torch.stack([y, cb, cr], dim=-3)


class YcbcrToRgb(nn.Module):
    r"""Convert image from YCbCr to RGB.

    The image data is assumed to be in the range of (0, 1).

    Args:
        input (torch.Tensor): YCbCr image to be converted to RGB. The image has to
        be in the shape of (*, 3, H, W).

    Return:
        torch.Tensor: RGB version of the image.
    """

    def __init__(self) -> None:
        super(YcbcrToRgb, self).__init__()

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # type: ignore
        return ycbcr_to_rgb(image)


def ycbcr_to_rgb(image: torch.Tensor) -> torch.Tensor:
    r"""Convert a YCbCr image to RGB.

    See :class:`~kornia.color.YcbcrToRgb` for details.

    Args:
        input (torch.Tensor): YCbCr Image to be converted to RGB.

    Returns:
        torch.Tensor: RGB version of the image.
    """

    if not torch.is_tensor(image):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W).Got {}"
                         .format(image.shape))

    # unpack the YCbCr values
    y, cb, cr = torch.chunk(image, dim=-3, chunks=3)

    # apply weights

    delta: float = 0.5
    cb_delta: torch.Tensor = cb - delta
    cr_delta: torch.Tensor = cr - delta

    r: torch.Tensor = y + 1.402000 * cr_delta
    g: torch.Tensor = y - 0.344136 * cb_delta - 0.714136 * cr_delta
    b: torch.Tensor = y + 1.772000 * cb_delta

    return torch.stack([r, g, b], dim=-3)
