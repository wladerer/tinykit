"""Shared helpers for POV-Ray based structure rendering (used by viz)."""

import contextlib
import json
import os
from pathlib import Path

import numpy as np
from ase.io import write

_ATOM_TEMPLATES_PATH = Path(__file__).parent / "resources" / "atom_templates.json"
_atom_templates = None


def available_cpus() -> int:
    """Number of CPUs this process may actually use.

    Prefers the affinity mask (respects cgroup/taskset limits) and falls back to
    the logical CPU count on platforms without `sched_getaffinity` (e.g. macOS).
    Always at least 1.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


@contextlib.contextmanager
def _chdir(path):
    """Temporarily run in `path`.

    ASE invokes `povray <name>.ini` with no working directory, and the .ini
    refers to the .pov/.png by basename. Rendering from the output file's own
    directory is what lets `-o subdir/x.png` (or any absolute path) work instead
    of only succeeding when the shell already sits in that directory.
    """
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _set_ini_threads(ini_path, n_threads: int) -> None:
    """Append `Work_Threads=N` to a POV-Ray .ini so it uses all available cores.

    ASE's .ini omits Work_Threads, and POV-Ray otherwise leaves cores idle on
    some builds. Appended after ASE writes the .ini and before the render runs.
    """
    if n_threads and os.path.exists(ini_path):
        with open(ini_path, "a") as fh:
            fh.write(f"\nWork_Threads={n_threads}\n")


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


def parse_rgb(spec) -> tuple:
    """Parse a color into a 0-1 RGB tuple.

    Accepts a '#rrggbb' hex string, a bare 'rrggbb' hex string, or a
    comma-separated triple. Comma triples are read as 0-255 if any component
    exceeds 1, otherwise taken as already on the 0-1 scale.
    """
    if isinstance(spec, (list, tuple)):
        vals = [float(x) for x in spec]
        return tuple(v / 255 for v in vals) if any(v > 1 for v in vals) else tuple(vals)
    spec = spec.strip()
    if spec.startswith('#') or (len(spec) == 6 and ',' not in spec):
        return tuple(c / 255 for c in hex_to_rgb(spec))
    vals = [float(x) for x in spec.split(',')]
    return tuple(v / 255 for v in vals) if any(v > 1 for v in vals) else tuple(vals)


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


def resolve_atom_styles(atoms, overrides: dict = None,
                        radius_scale: float = 1.0, radius_offset: float = 0.0):
    """Return per-atom (colors, radii) for an ASE Atoms object.

    Colors and radii come from the bundled atom templates (VESTA palette and
    covalent radii). `overrides` (an element -> style mapping, e.g. loaded from
    a YAML file) may replace the color and/or radius of any element without
    touching the shipped templates; see :func:`_parse_style_override` for the
    accepted per-element forms. Colors are given on the 0-255 scale (or hex) and
    radii on the template (covalent) scale.

    The final radius for each atom is ``template_radius * radius_scale +
    radius_offset``. Because the templates store covalent radii (where bonded
    atoms' spheres roughly touch at full scale), a `radius_scale` below 1 gives
    a ball-and-stick look that keeps binding geometry visible.
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
    radii = np.array([radius_map[a.symbol] for a in atoms]) * radius_scale + radius_offset
    return colors, radii


def add_render_args(parser, default_height: int = 900):
    """Add the common POV-Ray rendering controls to an argument parser."""
    group = parser.add_argument_group("rendering")
    group.add_argument('--radius-scale', type=float, default=0.6,
                       help='Ball-and-stick scale for atom radii relative to covalent '
                            'radii (default: 0.6; use 1.0 for near space-filling)')
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


def _parse_pov_atom_locs(pov_path):
    """Read the rendered atom centres (image-plane coords) and radii from a
    written .pov, keyed by the `// #N` atom index ASE appends to each line."""
    import re
    pat = re.compile(
        r"atom\(<\s*([-\d.]+),\s*([-\d.]+),\s*([-\d.]+)>,\s*([\d.]+),.*?// #(\d+)")
    locs, rads = {}, {}
    for line in Path(pov_path).read_text().splitlines():
        m = pat.search(line)
        if m:
            i = int(m.group(5))
            locs[i] = np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))])
            rads[i] = float(m.group(4))
    return locs, rads


def _arrow_pov(base, tip, shaft_r, head_r, head_frac, color, finish):
    """POV-Ray cylinder (shaft) + cone (head) for one arrow base->tip."""
    d = tip - base
    neck = base + (1.0 - head_frac) * d
    def v(a):
        return f"<{a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f}>"
    col = f"pigment {{ color rgb <{color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}> }}"
    fin = f"finish {{ {finish} }}"
    return (f"cylinder {{ {v(base)}, {v(neck)}, {shaft_r:.3f} {col} {fin} }}\n"
            f"cone {{ {v(neck)}, {head_r:.3f}, {v(tip)}, 0.0 {col} {fin} }}\n")


def _dashed_line_pov(base, tip, radius, dash_len, gap_len, color, finish):
    """POV-Ray dashed line: a row of short cylinders (with rounded caps) from
    base to tip, each `dash_len` long and separated by `gap_len`."""
    d = tip - base
    length = float(np.linalg.norm(d))
    if length < 1e-9 or dash_len <= 0:
        return ""
    u = d / length
    def v(a):
        return f"<{a[0]:.4f}, {a[1]:.4f}, {a[2]:.4f}>"
    col = f"pigment {{ color rgb <{color[0]:.3f}, {color[1]:.3f}, {color[2]:.3f}> }}"
    fin = f"finish {{ {finish} }}"
    period = dash_len + max(gap_len, 0.0)
    out, s = "", 0.0
    while s < length - 1e-6:
        a = base + u * s
        b = base + u * min(s + dash_len, length)
        out += (f"cylinder {{ {v(a)}, {v(b)}, {radius:.3f} {col} {fin} }}\n"
                f"sphere {{ {v(a)}, {radius:.3f} {col} {fin} }}\n"
                f"sphere {{ {v(b)}, {radius:.3f} {col} {fin} }}\n")
        s += period
    return out


def _render_with_geometry(
    atoms,
    output: str,
    geometry_fn,
    rotation=(0, 0, 0),
    colors=None,
    radii=None,
    povray_settings: dict = None,
    isosurface_data=None,
    cleanup: bool = True,
) -> str:
    """Render `atoms`, then splice extra POV-Ray geometry into the scene.

    ASE renders by first writing a `.pov`; this reads the rendered atom centres
    back out of that file and calls ``geometry_fn(locs, rads, scale, R)`` to
    build a POV-Ray source string that is appended before the image is
    produced. `locs`/`rads` map atom index -> image-plane centre / rendered
    radius; `scale` maps template-scale radii into the image frame (so dash and
    arrow sizes stay consistent with atom sizes); `R` is the rotation matrix
    ASE applied (for orienting direction vectors like moments).
    """
    from ase.io.utils import rotate
    pov_path = update_image_extension(output)
    rotation_str = array_to_rotation_string(rotation)
    renderer = write(
        pov_path, atoms, format="pov", rotation=rotation_str,
        colors=colors, radii=radii, povray_settings=povray_settings or {},
        isosurface_data=isosurface_data,
    )

    R = rotate(rotation_str)
    locs, rads = _parse_pov_atom_locs(pov_path)
    scale = 1.0
    if radii is not None:
        for i, rad in rads.items():
            if radii[i] > 1e-6:
                scale = rad / radii[i]
                break

    geometry = geometry_fn(locs, rads, scale, R)
    with open(pov_path, "a") as fh:
        fh.write("\n// tinykit appended geometry\n" + geometry)

    _set_ini_threads(pov_path.replace(".pov", ".ini"), available_cpus())
    with _chdir(os.path.dirname(os.path.abspath(pov_path))):
        image_path = renderer.render()
    if cleanup:
        for intermediate in (pov_path, pov_path.replace(".pov", ".ini")):
            if os.path.exists(intermediate):
                os.remove(intermediate)
    return image_path


def render_structure_with_moments(
    atoms,
    moments,
    output: str,
    rotation=(0, 0, 0),
    colors=None,
    radii=None,
    povray_settings: dict = None,
    cleanup: bool = True,
    length: float = 2.8,
    min_moment: float = 0.1,
    scale_by_magnitude: bool = False,
    shaft_r: float = 0.16,
    head_r: float = 0.40,
    head_frac: float = 0.34,
    up_color=(0.78, 0.20, 0.16),
    dn_color=(0.13, 0.24, 0.55),
    plane_color=(0.20, 0.55, 0.45),
    finish: str = "phong 0.9 phong_size 60 ambient 0.35",
) -> str:
    """Render an ASE Atoms object with per-atom magnetic-moment arrows.

    `moments` is an (natoms, 3) array of Cartesian moment vectors. Atoms whose
    moment magnitude is below `min_moment` get no arrow (this suppresses the
    near-zero residual moments that otherwise clutter a figure). Each drawn
    moment becomes a ray-traced arrow centred on its atom, coloured by
    orientation: +c (up_color), -c (dn_color), in-plane (plane_color).

    By default every arrow is drawn at the same `length` (direction only). With
    `scale_by_magnitude`, arrow length is proportional to |m| relative to the
    largest drawn moment, so relative moment sizes are visible. Arrows are
    placed in ASE's rotated/scaled image frame: atom centres are read back from
    the written .pov and the moment direction is rotated by the same matrix, so
    arrows track the atoms exactly.
    """
    moments = np.asarray(moments, dtype=float)
    norms = np.linalg.norm(moments, axis=1)
    cutoff = max(min_moment, 1e-3)
    drawn = norms >= cutoff
    ref = norms[drawn].max() if drawn.any() else 1.0

    def geometry(locs, rads, scale, R):
        arrows = ""
        for i in range(len(atoms)):
            n = float(norms[i])
            if n < cutoff or i not in locs:
                continue
            frac = (n / ref) if scale_by_magnitude else 1.0
            d_img = ((moments[i] / n * length * frac) @ R) * scale
            base, tip = locs[i] - 0.5 * d_img, locs[i] + 0.5 * d_img
            mz = moments[i][2] / n
            color = up_color if mz > 0.5 else dn_color if mz < -0.5 else plane_color
            arrows += _arrow_pov(base, tip, shaft_r * scale, head_r * scale,
                                 head_frac, color, finish)
        return arrows

    return _render_with_geometry(
        atoms, output, geometry, rotation=rotation, colors=colors,
        radii=radii, povray_settings=povray_settings, cleanup=cleanup)


def render_structure_with_bonds(
    atoms,
    bonds,
    output: str,
    rotation=(0, 0, 0),
    colors=None,
    radii=None,
    povray_settings: dict = None,
    isosurface_data=None,
    cleanup: bool = True,
    bond_color=(0.30, 0.30, 0.30),
    bond_radius: float = 0.10,
    dash_length: float = 0.30,
    gap_length: float = 0.22,
    finish: str = "phong 0.6 ambient 0.45",
) -> str:
    """Render an ASE Atoms object with dashed lines between atom-index pairs.

    `bonds` is an iterable of ``(i, j)`` zero-based atom-index pairs. Each pair
    is drawn as a dashed line connecting the two atoms' rendered centres, in the
    same image frame ASE uses, so the dashes track the atoms exactly. Out-of-
    range indices and pairs whose atoms were not rendered are skipped. The dash
    radius and lengths are given on the template radius scale and are mapped
    into the image frame, matching the atom sizing.

    This takes explicit pairs today; richer connectivity sources (COHP/COBI,
    bond length, magnetic coupling) can feed the same ``(i, j)`` list later.
    """
    bonds = list(bonds)
    n_atoms = len(atoms)

    def geometry(locs, rads, scale, R):
        out = ""
        for i, j in bonds:
            if not (0 <= i < n_atoms and 0 <= j < n_atoms):
                continue
            if i not in locs or j not in locs:
                continue
            out += _dashed_line_pov(
                locs[i], locs[j], bond_radius * scale,
                dash_length * scale, gap_length * scale, bond_color, finish)
        return out

    return _render_with_geometry(
        atoms, output, geometry, rotation=rotation, colors=colors,
        radii=radii, povray_settings=povray_settings,
        isosurface_data=isosurface_data, cleanup=cleanup)


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

    renderer = write(
        pov_path,
        atoms,
        format='pov',
        rotation=rotation_str,
        colors=colors,
        radii=radii,
        povray_settings=povray_settings or {},
        isosurface_data=isosurface_data,
    )
    _set_ini_threads(pov_path.replace('.pov', '.ini'), available_cpus())
    with _chdir(os.path.dirname(os.path.abspath(pov_path))):
        image_path = renderer.render()

    if cleanup:
        for intermediate in (pov_path, pov_path.replace('.pov', '.ini')):
            if os.path.exists(intermediate):
                os.remove(intermediate)

    return image_path
