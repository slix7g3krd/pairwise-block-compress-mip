import pytest
import graphics.block_compress_mip as mod

CASES = [
    ("border_extend", (1,)),
    ("alpha_mode", (1,)),
    ("mip_chain_order", (1,)),
    ("npot_normalize", (1,)),
    ("mip_from_source", (1,)),
    ("block_deterministic", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("encode_block", (1,)),
    ("metrics", (1,)),
    ("mip_level", (1,)),
    ("sample_uv", (1,)),
    ("compress", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
