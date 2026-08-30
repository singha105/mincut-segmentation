"""Opening images, including HEIC/HEIF straight off an iPhone.

Pillow has no built-in HEIC decoder, so uploads from Apple devices used to be
rejected outright and had to be converted to JPEG or PNG first. `pillow-heif`
plugs a decoder into Pillow, after which `Image.open()` handles .heic the same
as anything else.

The plugin is registered once, lazily, on the first open. If it is missing the
rest of the project still works and only HEIC files fail, with a message saying
what to install rather than a decoder traceback.
"""

from PIL import Image, ImageOps

# Formats the app and CLI accept. Streamlit wants them without the leading dot.
UPLOAD_TYPES = ["png", "jpg", "jpeg", "heic", "heif"]

_HEIF_REGISTERED = False
_HEIF_ERROR = None


def register_heif():
    """Register the HEIF decoder with Pillow. Returns True when available.

    Safe to call repeatedly; the work happens once.
    """
    global _HEIF_REGISTERED, _HEIF_ERROR
    if _HEIF_REGISTERED:
        return True
    if _HEIF_ERROR is not None:
        return False
    try:
        import pillow_heif
    except ImportError as error:  # pragma: no cover - depends on the environment
        _HEIF_ERROR = error
        return False

    pillow_heif.register_heif_opener()
    _HEIF_REGISTERED = True
    return True


def heif_available():
    """Whether HEIC/HEIF uploads can be decoded in this environment."""
    return register_heif()


def looks_like_heif(name):
    """Guess from a filename whether this is a HEIC/HEIF file."""
    return bool(name) and str(name).lower().rsplit(".", 1)[-1] in {"heic", "heif", "hif"}


def open_image(source, name=None):
    """Open `source` as RGB, or raise ValueError with something readable.

    `source` is a path or file-like object. `name` is the original filename
    when `source` does not carry one, used only to give a better error.

    Two things happen beyond a plain Image.open:

    - HEIF support is registered first, so iPhone photos open directly.
    - EXIF orientation is applied. Phone cameras record rotation in metadata
      rather than rotating the pixels, so without this a photo taken sideways
      would be segmented sideways.
    """
    register_heif()

    try:
        image = Image.open(source)
        image.load()
    except Exception as error:
        filename = name or getattr(source, "name", None) or str(source)
        if looks_like_heif(filename) and not heif_available():
            raise ValueError(
                "HEIC/HEIF support needs the pillow-heif package. "
                "Install it with: pip install -r requirements.txt"
            ) from error
        raise ValueError(
            f"Couldn't read that file as an image ({type(error).__name__}: {error})."
        ) from error

    return ImageOps.exif_transpose(image).convert("RGB")
