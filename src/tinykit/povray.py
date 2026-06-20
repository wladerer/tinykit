"""Shared helpers for POV-Ray based structure rendering (slabviz, bulkviz)."""

import json
import os
from pathlib import Path

import numpy as np
from ase.io import write

_ATOM_TEMPLATES_PATH = Path(__file__).parent / "resources" / "atom_templates.json"
_atom_templates = None


def array_to_rotation_string(array) -> str:
    """Convert a 3-element rotation array into ASE's POV rotation string."""
    return f"{array[0]}x,{array[1]}y,{array[2]}z"


def update_image_extension(string: str) -> str:
    """Force an output filename to the .pov extension expected by ASE's writer.

    ASE renders by writing a .pov (plus a .ini), then producing the final image.
    A name without an extension gets .pov appended; a .png name is rewritten to
    .pov; a name already ending in .pov is returned unchanged.
    """
    if not string.endswith('.pov') and '.' not in string:
        return f'{string}.pov'

    if not string.endswith('.pov'):
        return string.replace('.png', '.pov')

    return string


def hex_to_rgb(hex_string: str) -> tuple:
    """Convert a '#rrggbb' string to an (r, g, b) tuple on the 0-255 scale."""
    h = hex_string.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def normalize_colors(color_map: dict) -> dict:
    """Scale an element->RGB(0-255) map down to the 0-1 range POV-Ray expects."""
    return {k: tuple(c / 255 for c in v) for k, v in color_map.items()}


def _load_atom_templates() -> dict:
    global _atom_templates
    if _atom_templates is None:
        _atom_templates = json.loads(_ATOM_TEMPLATES_PATH.read_text())
    return _atom_templates


def default_color_map() -> dict:
    """Element -> RGB(0-255) colors from the bundled atom templates."""
    return {k: hex_to_rgb(v['color']) for k, v in _load_atom_templates().items()}


def default_radius_map() -> dict:
    """Element -> rendering radius from the bundled atom templates."""
    return {k: v['radius'] for k, v in _load_atom_templates().items()}


def _parse_style_override(spec):
    """Split a single element's style override into (color, radius).

    Accepts three forms (any field may be omitted):
        Fe: [255, 100, 0]                      # color only (RGB 0-255)
        Fe: "#ff6400"                          # color only (hex)
        Fe: {color: [255, 100, 0], radius: 1.3}  # color and/or radius
    """
    if isinstance(spec, dict):
        color = spec.get('color')
        radius = spec.get('radius')
    elif isinstance(spec, (list, tuple, str)):
        color, radius = spec, None
    else:
        raise ValueError(f"Invalid style override {spec!r}; expected RGB list, hex string, or mapping")

    if isinstance(color, str):
        color = hex_to_rgb(color)
    return color, radius


def resolve_atom_styles(atoms, overrides: dict = None, radius_offset: float = 0.0):
    """Return per-atom (colors, radii) for an ASE Atoms object.

    Colors and radii come from the bundled atom templates. `overrides` (an
    element -> style mapping, e.g. loaded from a YAML file) may replace the
    color and/or radius of any element without touching the shipped templates;
    see :func:`_parse_style_override` for the accepted per-element forms. Colors
    are given on the 0-255 scale (or hex) and radii on the template scale; the
    final radii are shifted by `radius_offset`.
    """
    color_map = default_color_map()
    radius_map = default_radius_map()

    for element, spec in (overrides or {}).items():
        color, radius = _parse_style_override(spec)
        if color is not None:
            color_map[element] = tuple(color)
        if radius is not None:
            radius_map[element] = radius

    color_map = normalize_colors(color_map)
    colors = [color_map[a.symbol] for a in atoms]
    radii = np.array([radius_map[a.symbol] for a in atoms]) + radius_offset
    return colors, radii


def add_render_args(parser, default_height: int = 900):
    """Add the common POV-Ray rendering controls to an argument parser."""
    group = parser.add_argument_group("rendering")
    group.add_argument('--width', type=int, default=None,
                       help='Canvas width in pixels (default: auto from height)')
    group.add_argument('--height', type=int, default=default_height,
                       help=f'Canvas height in pixels (default: {default_height})')
    group.add_argument('--camera-dist', type=float, default=20.0,
                       help='Distance from camera to the front atom (default: 20)')
    camera = group.add_mutually_exclusive_group()
    camera.add_argument('--orthographic', dest='camera_type', action='store_const', const='orthographic',
                        help='Orthographic camera (default)')
    camera.add_argument('--perspective', dest='camera_type', action='store_const', const='perspective',
                        help='Perspective camera')
    group.set_defaults(camera_type='orthographic')
    group.add_argument('--show-cell', action='store_true', help='Draw the unit cell edges')
    group.add_argument('--keep-pov', action='store_true',
                       help='Keep the intermediate .pov/.ini files instead of deleting them')
    return parser


def povray_settings_from_args(args, extra: dict = None) -> dict:
    """Build an ASE povray_settings dict from parsed --width/--height/etc.

    ASE forbids setting canvas width and height simultaneously, so width takes
    precedence when given and height is used otherwise.
    """
    if args.width is not None:
        canvas_width, canvas_height = args.width, None
    else:
        canvas_width, canvas_height = None, args.height

    settings = {
        'canvas_width': canvas_width,
        'canvas_height': canvas_height,
        'camera_dist': args.camera_dist,
        'camera_type': args.camera_type,
        'celllinewidth': 0.05 if args.show_cell else 0.0,
    }
    if extra:
        settings.update(extra)
    return settings


def render_structure(
    atoms,
    output: str,
    rotation=(0, 0, 0),
    colors=None,
    radii=None,
    povray_settings: dict = None,
    isosurface_data=None,
    cleanup: bool = True,
) -> str:
    """Render an ASE Atoms object to an image with POV-Ray.

    Writes a .pov/.ini pair, renders the image, and (when `cleanup` is True)
    removes the intermediate .pov and .ini files. Returns the image path.
    """
    pov_path = update_image_extension(output)
    rotation_str = array_to_rotation_string(rotation)

    image_path = write(
        pov_path,
        atoms,
        format='pov',
        rotation=rotation_str,
        colors=colors,
        radii=radii,
        povray_settings=povray_settings or {},
        isosurface_data=isosurface_data,
    ).render()

    if cleanup:
        for intermediate in (pov_path, pov_path.replace('.pov', '.ini')):
            if os.path.exists(intermediate):
                os.remove(intermediate)

    return image_path
