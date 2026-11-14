import importlib.util
import sys
import types
# Tooling-only shim: this container lacks pycocotools. User's YOLO environment
# has it installed; tests here do not decode any COCO masks.
fake = types.ModuleType('pycocotools')
fake.mask = types.ModuleType('pycocotools.mask')
sys.modules.setdefault('pycocotools', fake)
sys.modules.setdefault('pycocotools.mask', fake.mask)
import pathlib
import numpy as np

modpath = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate_refinement_oof.py'
spec = importlib.util.spec_from_file_location('v112', modpath)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_refinement():
    a=np.zeros((50,50),dtype=bool); a[22:25,8:35]=True
    dark=np.ones((50,50),dtype=np.float32)*5
    for variant in m.VARIANTS:
        b=m.refine_one(a,dark,variant)
        assert b.shape == a.shape
        assert b.dtype==bool
        assert b.any()
    assert m.refine_one(a,dark,'baseline').sum()==a.sum()
    assert m.refine_one(a,dark,'grow1').sum()>=a.sum()
    x=np.zeros((50,50,3),dtype=np.uint8)
    f=[(a,.9),(np.roll(a,15,axis=0),.8)]
    for variant in m.VARIANTS:
        res=m.refine_final(f,x,variant)
        assert len(res)==len(f)
        assert not np.logical_and(*res).any()
    print('PASS: four variants, baseline unchanged, nonoverlap preserved')

if __name__=='__main__':
    test_refinement()
