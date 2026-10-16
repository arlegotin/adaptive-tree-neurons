"""Declared implementation capabilities; verification status belongs to each run.

An installed dependency is not a verified adapter. Optional assets are reasoned
exclusions; required pending recipes keep the corresponding track gate open.
"""
from dataclasses import asdict, dataclass
from importlib.metadata import version


@dataclass(frozen=True)
class AdapterCapability:
    method: str
    version: str
    status: str
    track: str
    required: bool
    recipe: str
    source: str
    preprocessing: str
    categories: str
    context: str | bool
    internal_selection: str | bool
    transduction: bool | None
    cache: bool | None
    backend: str
    fallback: bool | None
    unavailable_reason: str | None
    unknown_counts_reason: str


def capabilities() -> list[AdapterCapability]:
    """Describe supported paths, without claiming that this call executed them."""
    common = dict(version=version('treeneuron'),status='implemented',track='causal',required=True,
        source='local treeneuron package; exact source hash in verification receipt',
        preprocessing='F-only transform',categories='F-only ordinal codes and missing indicators',
        context=False,internal_selection=False,transduction=False,cache=False,
        backend='numpy CPU float64',fallback=False,unavailable_reason=None,
        unknown_counts_reason='parameter/serialized counts require a fitted instance; process memory not measured by registry')
    result = [AdapterCapability(method=m,recipe=r,**common) for m,r in
              [('Mean','F target mean'),('Ridge','linear ridge; caller selects alpha with V')]]
    result.append(AdapterCapability(**{**common,'method':'SplineRidge',
        'recipe':'additive cubic regression spline ridge; caller selects alpha with V',
        'categories':'unsupported; explicit error','preprocessing':'F-only numeric imputation/scaling and per-feature quantile knots'}))
    for method,recipe in [('B1','continued ordinary keep'),('B2','ordinary width brackets and zero-head widening; duplication deferred'),
        ('B3','static partial/full centered/direct-leaf coordinates'),('B4','matched random allocation'),
        ('B5','Adam/SGD eigenvector allocation and top-ID Gaussian directions'),
        ('B6','ANT-inspired restricted independent single-unit G probes; fixed count; fresh joint continuation')]:
        result.append(AdapterCapability(**{**common,'method':method,'recipe':recipe,
            'context':'original DevelopmentData F/G provenance; no V/T access',
            'internal_selection':'G checkpoint selection; G allocation for B6','backend':'PyTorch CPU eager float32; CPU64 scoring',
            'categories':'numerical causal fixtures only; transformed width must equal declared model width'}))
    for method,track,required,reason in [
        ('B7','causal',True,'Evidence Task3: pinned official NODE recipe not implemented or verified'),
        ('B8','practical',True,'RealMLP-TD: pinned pytabkit native recipe and preprocessing pending'),
        ('B9','practical',True,'TabM: default/embedding k=32 native recipe and member-loss contract pending'),
        ('B10','practical',True,'CatBoost: native adapter and categorical/missingness contract pending'),
        ('B14','practical',True,'LightGBM: native constant-leaf/linear-tree adapter and preprocessing pending'),
        ('B11','practical',True,'XGBoost: pinned CPU histogram adapter and thread contract pending'),
        ('B13','practical',False,'TabPFN-3.5: optional pinned model assets, license and adapter unavailable'),
        ('B12','practical',False,'TabICLv2: optional pinned model assets, license and adapter unavailable')]:
        result.append(AdapterCapability(**{**common,'method':method,'track':track,'required':required,
            'status':'pending' if required else 'unavailable','version':'unverified; no pinned adapter',
            'recipe':'pending verification','source':'not implemented',
            'preprocessing':'unverified','categories':'unverified','context':'unverified',
            'internal_selection':'unverified','transduction':None,'cache':None,'backend':'unverified',
            'fallback':None,'unavailable_reason':reason}))
    return result


def collect_practical_report():
    matrix = [asdict(c) for c in capabilities() if c.track == 'practical']
    return dict(schema_version=1,track='practical',implemented_checks_passed=False,
                track_ready=False,adapter_matrix=matrix,
                reason='Required practical adapters are pending; optional assets are explicit exclusions')
