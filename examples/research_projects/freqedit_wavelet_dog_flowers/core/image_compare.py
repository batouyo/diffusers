from pathlib import Path
from typing import List, Sequence, Union

from PIL import Image


ImageInput = Union[str, Path, Image.Image]


def _load_rgb_images(images: Sequence[ImageInput]) -> List[Image.Image]:
    rgb_images = []
    for item in images:
        if isinstance(item, Image.Image):
            rgb_images.append(item.convert("RGB").copy())
        else:
            with Image.open(item) as img:
                rgb_images.append(img.convert("RGB").copy())
    return rgb_images


def _pad_to_canvas(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    if image.size == size:
        return image

    canvas = Image.new("RGB", size, color=(0, 0, 0))
    x = (size[0] - image.width) // 2
    y = (size[1] - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def save_compare_artifacts(
    images: Sequence[ImageInput],
    output_dir: Union[str, Path],
    png_name: str = "compare.png",
    gif_name: str = "compare.gif",
    duration: int = 600,
) -> List[str]:
    if not images:
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rgb_images = _load_rgb_images(images)
    max_width = max(img.width for img in rgb_images)
    max_height = max(img.height for img in rgb_images)

    frame_size = (max_width, max_height)
    frames = [_pad_to_canvas(img, frame_size) for img in rgb_images]

    compare_png = output_path / png_name
    total_width = sum(frame.width for frame in frames)
    concat = Image.new("RGB", (total_width, max_height), color=(0, 0, 0))
    x_offset = 0
    for frame in frames:
        concat.paste(frame, (x_offset, 0))
        x_offset += frame.width
    concat.save(compare_png)

    saved_paths = [str(compare_png)]

    if frames:
        compare_gif = output_path / gif_name
        frames[0].save(
            compare_gif,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
        saved_paths.append(str(compare_gif))

    return saved_paths
