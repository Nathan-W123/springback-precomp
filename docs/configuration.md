# Input-deck reference

A run of `sparlab_form` is fully determined by one JSON deck. Every numeric
field is SI (`docs/conventions.md`). Unknown keys are **reported**, and with
`--strict-config` they are an error - a misspelled key that silently takes its
default is one of the easiest ways to publish a wrong number.

Two conveniences beyond strict JSON: `//` and `/* */` comments, and a trailing
comma before `}` or `]`. Everything else follows RFC 8259, and a syntax error
reports its line and column.

`sparlab_form` runs the analysis of the deck's `forming` block and refuses a
deck without one. The analysis applies no load case: the tools and the
prescribed displacements drive the model.

## Top level

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `name` | string | `"case"` | case name; also the default output directory, `results/<name>` |
| `description` | string | `""` | free text, echoed into `summary.json` |
| `units` | string | `"SI"` | informational only |
| `mesh` | object | required | see below |
| `material` | object | required | see below |
| `material_regions` | array | `[]` | other materials on element regions; see below |
| `model` | object | `{}` | element formulation and integration |
| `boundary_conditions` | array | `[]` | displacement constraints, taken by every step that lists none of its own; may be omitted when every step lists its own |
| `forming` | object | required | incremental forming by rigid tools; its keys, defaults and refusals are in [`forming.md`](forming.md), section 2, and the `explicit` block of a `form_explicit` step in section 7.2 |

## `mesh`

```json
"mesh": { "type": "structured_hex", "nx": 20, "ny": 20, "nz": 2,
          "lx": 0.04, "ly": 0.04, "lz": 0.001, "x0": -0.02, "y0": -0.02, "z0": -0.001 }

"mesh": { "type": "file", "path": "blank.inp", "scale": 0.001 }
```

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `type` | string | required | `structured_hex` (a box of Hex8 cells) or `file` (read from a mesh file); there is no default |
| `nx`, `ny`, `nz` | integer | required for `structured_hex` | elements per direction, `>= 1` |
| `lx`, `ly`, `lz` | number | required for `structured_hex` | domain extents [m], `> 0` |
| `x0`, `y0`, `z0` | number | `0` | lower corner [m] |

The mesh is solid: three displacement components per node, three
components for every point and vector of the deck, and boundary conditions
may fix `"z"`.

**Meshes read from a file** (`"type": "file"`)

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `path` | string | required | the mesh file; a relative path resolves against the deck's directory |
| `format` | string | `auto` | `auto` (from the extension: `.inp` is Abaqus / CalculiX) or `abaqus` |
| `scale` | number | `1` | factor applied to every coordinate: `0.001` for a mesh drawn in millimetres |
| `merge_duplicate_nodes` | bool | `false` | merge coincident nodes instead of only reporting them |
| `duplicate_tolerance` | number [m] | `0` | distance below which two nodes coincide, after scaling; `0` means `1e-9` times the bounding-box diagonal |

Abaqus / CalculiX `.inp` files are read, `*INCLUDE`d files with them. The
cells of the highest dimension in the file become the mesh, and they must be
linear hexahedra (`C3D8` and its variants); a file of tetrahedra, prisms,
pyramids or second-order cells, or one that mixes cell types, is refused
with a message saying how to re-export it. Lower dimensional elements (the
boundary facets a mesher writes for its groups) only feed the named sets: an
`*NSET` is a node set, and an `*ELSET` is an element set when it holds cells
and a node set when it holds boundary elements. Mirrored cells are re-ordered
and counted, nodes no cell uses are dropped, coincident nodes are reported,
and a domain larger than 20 m or smaller than 0.1 mm draws a warning about
units. Boundary conditions, loads and materials in an `.inp` file are not
imported, and the reader warns when the file has them. The reader logs what
it read and did; `summary.json` records the mesh under `mesh.source`.

The resolution of a file mesh comes from the mesher, so `nx` ... `z0` are
errors on a `file` mesh.

## `material`

```json
"material": { "name": "AA1050-like", "youngs_modulus": 70.0e9,
              "poisson_ratio": 0.33, "density": 2700.0 }
```

| Key | Type | Default | Constraint |
|-----|------|---------|------------|
| `name` | string | `"material"` | names the material in messages |
| `youngs_modulus` | number [Pa] | required | `> 0` |
| `poisson_ratio` | number | required | in `(-1, 0.5)` so the constitutive matrix stays positive definite |
| `density` | number [kg/m^3] | `0` | `>= 0`; a `form_explicit` step needs a positive density of every material |
| `thermal_expansion`, `reference_temperature`, `conductivity` | number | `0` | accepted and checked (finite; `conductivity >= 0`), without effect: the forming analysis applies no temperature |
| `plasticity` | object | none | plasticity, below; without it the material is elastic |

### `material.plasticity`

```json
"plasticity": { "yield_stress": 250e6, "hardening_modulus": 2e9,
                "saturation_stress": 80e6, "saturation_rate": 25,
                "kinematic_hardening_modulus": 3e9 }
```

Rate-independent plasticity of the forming analysis (`docs/formulation.md`,
section 7): von Mises (J2) or Hill's 1948 anisotropic yield criterion, with
isotropic and kinematic hardening. The yield stress after an accumulated
plastic strain `alpha` is
`sigma_y(alpha) = yield_stress + hardening_modulus alpha + saturation_stress (1 - exp(-saturation_rate alpha))`
(linear isotropic hardening plus Voce saturation), and the yield surface
moves with the back stress of Prager's linear kinematic hardening,
`d beta = (2/3) kinematic_hardening_modulus d eps_p`, and/or of Chaboche
backstresses (below). In uniaxial tension the
plastic slope of the stress against the plastic strain is
`hardening_modulus + kinematic_hardening_modulus` (plus the Voce part).

| Key | Type | Default | Constraint |
|-----|------|---------|------------|
| `yield_stress` | number [Pa] | required | `> 0`: the initial uniaxial yield stress `sigma_y0` (with Hill48, the equivalent stress at yield: the uniaxial yield stress along the rolling direction for the r-value and stress-ratio calibrations) |
| `hardening_modulus` | number [Pa] | `0` | `>= 0`, `H`: linear isotropic hardening (`0` with the others `0` is perfect plasticity) |
| `saturation_stress` | number [Pa] | `0` | `>= 0`, `Q`: the Voce saturation increment of the yield stress |
| `saturation_rate` | number [-] | `0` | `> 0` when `saturation_stress` is given, `delta` |
| `kinematic_hardening_modulus` | number [Pa] | `0` | `>= 0`, `H_kin`: Prager's linear kinematic hardening |
| `yield_criterion` | string | `von_mises` | `von_mises` or `hill48`; `hill48` needs the `anisotropy` block and the block needs `hill48` |
| `anisotropy` | object | - | the Hill48 calibration and material frame (below) |
| `backstresses` | array | `[]` | at most 4 Armstrong-Frederick backstresses `{ "modulus": C_i, "recovery": gamma_i }` (below) |
| `kinematic_integration` | string | `exponential` | how a backstress is integrated over a step: `exponential` or `backward_euler` |

**Hill48 anisotropy** (`"yield_criterion": "hill48"`). Sheet metal rolled
along RD, with transverse direction TD and normal ND, yields when
`F (s_TT - s_NN)^2 + G (s_NN - s_RR)^2 + H (s_RR - s_TT)^2 + 2 L s_TN^2 + 2 M s_NR^2 + 2 N s_RT^2 = sigma_y(alpha)^2`
of the relative stress in that frame, with associative (isochoric) flow.

```json
"plasticity": { "yield_stress": 170e6, "saturation_stress": 90e6, "saturation_rate": 12,
  "yield_criterion": "hill48",
  "anisotropy": { "r0": 1.9, "r45": 1.5, "r90": 2.3, "out_of_plane_shear": [1.5, 1.5],
                  "rolling_direction": [1, 0, 0], "sheet_normal": [0, 0, 1] },
  "backstresses": [ { "modulus": 60e9, "recovery": 600 }, { "modulus": 8e9, "recovery": 60 },
                    { "modulus": 1e9, "recovery": 0 } ],
  "kinematic_integration": "exponential" }
```

| `anisotropy` key | Type | Default | Meaning |
|------------------|------|---------|---------|
| `r0`, `r45`, `r90` | numbers [-] | - | Lankford coefficients at 0, 45 and 90 deg to RD, `> 0`: `F = r0 / (r90 (1 + r0))`, `G = 1 / (1 + r0)`, `H = r0 / (1 + r0)`, `N = (r0 + r90)(1 + 2 r45) / (2 r90 (1 + r0))`, so that `sigma_y` is the yield stress along RD |
| `stress_ratios` | object | - | `{ "sigma_45": s45, "sigma_90": s90, "sigma_biaxial": sb }`, the yield stresses at 45 and 90 deg and in equibiaxial tension over the one along RD, `> 0`: `G + H = 1`, `F + H = 1 / s90^2`, `F + G = 1 / sb^2`, `2 N = 4 / s45^2 - F - G` |
| `coefficients` | object | - | `{ "F", "G", "H", "L", "M", "N" }` directly (the yield stress along RD is then `yield_stress / sqrt(G + H)`) |
| `out_of_plane_shear` | `[L, M]` | `[1.5, 1.5]` | the out-of-plane shear coefficients of the r-value and stress-ratio calibrations (in-plane tests do not measure them; `1.5` is isotropic) |
| `rolling_direction` | `[x, y, z]` | `[1, 0, 0]` | RD in global axes, projected normal to ND |
| `sheet_normal` | `[x, y, z]` | `[0, 0, 1]` | ND in global axes; TD = ND x RD |
| `rolling_angle` | number [deg] | - | RD in the x-y plane at this angle from x, ND = z (instead of the two vectors) |

Exactly one of the r-values, `stress_ratios` and `coefficients` is given
(`r = 1`, or all ratios `1`, is von Mises). The coefficients must make a
convex surface: `L, M, N > 0`, `F + H > 0` and `FG + GH + HF > 0`. The frame
directions always have three components. With finite kinematics the frame
is the reference one and turns with the material. Hill48 in uniaxial
tension at `theta` from RD yields at
`sigma_y [F sin^4 + G cos^4 + H cos^2 2theta + 2 N sin^2 cos^2]^(-1/2)` with the
r-value `[H + (2N - F - G - 4H) sin^2 cos^2] / [F sin^2 + G cos^2]` - both
checked to round-off by the unit tests.

**Chaboche backstresses** (`backstresses`). The back stress is the sum of up
to four Armstrong-Frederick terms,
`d alpha_i = (2/3) C_i d eps_p - gamma_i alpha_i d alpha` (with Prager's
`kinematic_hardening_modulus` as one more, without recovery - at most four in
all), each saturating at `C_i / gamma_i` in uniaxial tension - for von
Mises, `sigma = sigma_y(alpha) + sum_i (C_i / gamma_i)(1 - exp(-gamma_i alpha))`
on a monotonic path (with Hill48 the backstress grows along the Hill flow
direction, not along the stress, and a uniaxial path is not proportional).
`modulus` is required and `> 0`, `recovery` defaults to `0` (Prager) and is
`>= 0`. With any `recovery > 0` the consistent tangent is not symmetric, and
the forming analysis factorises it by LU (`docs/forming.md`, section 1.5).
`exponential` integrates each backstress exactly for a flow direction fixed
over the step (so a uniaxial path is exact at any step size);
`backward_euler` is first-order accurate.

## `material_regions`

```json
"material_regions": [
  { "name": "aluminium_half", "region": { "box": { "xmin": 0.1 } },
    "material": { "name": "aluminium", "youngs_modulus": 70.0e9, "poisson_ratio": 0.33,
                  "density": 2700.0 } }
]
```

Each entry gives the elements of `region` (selected at their centroids, like
every element region) the `material`, which takes the same keys as the top-level
`material`. Later entries override earlier ones on shared elements; every
other element keeps the top-level material. A region that selects no element
is an error. Every evaluation of an element - its stiffness and internal
forces, its plasticity, its mass - uses the element's own material.

## `model`

A sheet of solid Hex8, one element layer through its thickness along `z`:

```json
"model": { "element_formulation": "incompatible_modes",
           "integration": { "thickness_points": 5, "thickness_direction": "z" } }
```

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `element_formulation` | string | `standard` | `standard`: the trilinear Hex8; `incompatible_modes`: the Hex8 with Wilson-Taylor incompatible modes (nine condensed parameters, Taylor's correction), which does not lock in bending - one or two elements through a sheet's thickness bend correctly. Every kinematics and material of the forming analysis (elastic, neo-Hookean, J2, Hill48, Chaboche); not with `mean_dilatation: all` (`auto` leaves it off for these elements) (`docs/formulation.md`, "The Hex8 with incompatible modes") |
| `integration.stiffness_points` | integer | `2` | Gauss points per direction for the stiffness and the internal forces, 1-4 (2 x 2 x 2) |
| `integration.mass_points` | integer | `3` | Gauss points per direction for the consistent element mass, 1-4, whose row sums are the lumped mass of a `form_explicit` step |
| `integration.thickness_points` | integer | `0` (= `stiffness_points`) | Gauss points through the thickness of a sheet, 1-7, for the stiffness rule `stiffness_points` x `stiffness_points` x `thickness_points` (standard or incompatible modes): 5 to 7 integrate the kinked elastic-plastic stress of a bent sheet, which springback is sensitive to. The plastic history, the stress points and the outputs follow the rule. With `element_formulation: incompatible_modes` a one-point rule (here or in `stiffness_points`) is refused: the mode gradients vanish at the element centre |
| `integration.thickness_direction` | string | `z` | `x`, `y` or `z`: the direction through the sheet, which carries `thickness_points` - the element's natural axis that the structured generator aligns with it (elements of a mesh from a file whose axis across the sheet is another are warned about) |
| `integration.face_points` | integer | `2` | accepted (`edge_points` as a synonym) and without effect: the forming analysis integrates no load over faces |
| `stress_state` | string | `three_dimensional` | `three_dimensional` (or `3d`), the only value a Hex8 mesh accepts |
| `thickness` | number [m] | `1.0` | a solid mesh has no thickness: any other value is refused |

## Regions

Boundary conditions, material regions and tool surfaces take a `region`
object. A region is the union of one or more primitives, optionally
complemented.

```json
"region": { "name": "frame", "invert": false, "tolerance": 0.0,
            "any_of": [
              { "box": { "xmax": -0.015 } }, { "box": { "xmin": 0.015 } },
              { "box": { "ymax": -0.015 } }, { "box": { "ymin": 0.015 } }
            ] }
```

A single primitive may also be written directly in the `region` object, without
`any_of`.

| Primitive | Form | Selects |
|-----------|------|---------|
| `all` | `"all": true` | everything |
| `box` | `{ "xmin":, "xmax":, "ymin":, "ymax":, "zmin":, "zmax": }`, any subset | points inside the axis-aligned box; omitted sides are unbounded |
| `circle` | `{ "center": [x, y, z], "radius": r, "axis": "z" }` | the infinite cylinder of that radius about the line through `center` along `axis` (`x`, `y` or `z`, default `z`) |
| `annulus` | `{ "center": [x, y, z], "inner_radius": r0, "radius": r1, "axis": "z" }` | the closed tube between the two radii, `r0 < r1` |
| `sphere` | `{ "center": [x, y, z], "radius": r }` | the closed ball |
| `node_ids` | `[...]` | explicit nodes (node regions only) |
| `element_ids` | `[...]` | explicit elements (element regions only) |
| `nearest_node` | `[x, y, z]` | the single node closest to the point (node regions only) |
| `group` | `"name"` | a named set of a mesh read from a file: its node set in a node region, its element set in an element region |

A `group` that the mesh does not have is an error listing the sets it does
have; so is a `group` on a structured mesh, which has none.

| Region key | Type | Default | Meaning |
|------------|------|---------|---------|
| `name` | string | auto | used in diagnostics |
| `invert` | bool | `false` | select the complement of the union |
| `tolerance` | number [m] | `0` | absolute geometric tolerance; `0` means `1e-9` times the mesh diagonal |

A degenerate box such as `{"xmax": 0.0}` reliably captures the plane of nodes
at `x = 0` because of that relative tolerance. Element regions are tested at
the element centroid. Naming two primitives in one region object without
`any_of` is an error, as is a region that selects nothing.

## `boundary_conditions`

```json
"boundary_conditions": [
  { "name": "clamp", "fix": ["x", "y", "z"], "value": [0.0, 0.0, 0.0],
    "region": { "box": { "xmax": 0.0 } } }
]
```

| Key | Type | Default | Meaning |
|-----|------|---------|---------|
| `name` | string | auto | diagnostics |
| `fix` | array of `"x"` / `"y"` / `"z"` | required, non-empty | which components to prescribe |
| `value` | `[u_x, u_y, u_z]` [m] | zeros | prescribed displacement |
| `region` | object | required | node region |

A step that lists no `boundary_conditions` of its own takes these, in
`absolute` mode: each prescribed component moves linearly over the step to
its `value` (`docs/forming.md`, section 1.1). A step's own constraints take
the same keys plus `mode`. Prescribed DOFs are removed from the system by
partitioning (`docs/formulation.md`, section 4), so the reactions are exact.

## `forming`

The tools, the steps, the Newton settings and the output switches of the
analysis: [`forming.md`](forming.md), section 2 lists every key with its
default and the refusals, and section 7.2 the `explicit` block of a
`form_explicit` step. `summary.json`, `config.json`, `mesh.json` and
`tool_forces.csv` are always written; `output.vtk` and `output.snapshots`
select the rest (section 3).

## Command line

```
sparlab_form --config <deck.json> [--output <dir>] [--threads <n>] [--no-vtk]
             [--strict-config] [--verbosity trace|debug|info|warn|error|silent]
             [--version] [--help]
```

`--output` defaults to `results/<name>`; `--threads` sets the OpenMP threads
of the element loops (default: all); `--no-vtk` skips the per-step VTK files
whatever `output.vtk` says; `--strict-config` makes unknown keys errors.
`sparlab_form --help` prints the list.
