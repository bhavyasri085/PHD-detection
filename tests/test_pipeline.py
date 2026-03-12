import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch, numpy as np


def test_backbone():
    print("backbone...", end=" ")
    from src.backbone.cspdarknet import build_backbone
    m = build_backbone("s")
    x = torch.randn(2,3,480,640)
    o = m(x)
    assert o["dark3"].shape == (2,128,60,80)
    assert o["dark4"].shape == (2,256,30,40)
    assert o["dark5"].shape == (2,512,15,20)
    print("✅")

def test_neck():
    print("neck...", end=" ")
    from src.backbone.cspdarknet import build_backbone
    from src.neck.panet import YoloPAFPN
    m = build_backbone("s"); n = YoloPAFPN("s")
    o = n(m(torch.randn(2,3,480,640)))
    assert len(o)==3
    print("✅")

def test_model_train():
    print("model train...", end=" ")
    from src.models.yolox import build_model
    m = build_model("s"); m.train()
    o = m(torch.randn(2,3,480,640))
    assert len(o)==3
    print("✅")

def test_model_infer():
    print("model infer...", end=" ")
    from src.models.yolox import build_model
    m = build_model("s"); m.eval()
    with torch.no_grad():
        dets = m(torch.randn(2,3,480,640))
    assert len(dets)==2
    print("✅")

def test_loss():
    print("loss...", end=" ")
    from src.models.yolox import build_model
    from src.loss.yolox_loss import YoloXLoss
    m = build_model("s"); m.train()
    c = YoloXLoss(1, img_size=(480,640))
    t = torch.tensor([[0.,0.,0.5,0.4,0.1,0.12],[1.,0.,0.3,0.5,0.08,0.09]])
    loss, _ = c(m(torch.randn(2,3,480,640)), t)
    assert loss.item() > 0
    print("✅")

def test_metrics():
    print("metrics...", end=" ")
    from src.utils.metrics import DetectionEvaluator
    e = DetectionEvaluator()
    gt = np.array([[10,10,50,50],[60,60,100,100]],dtype=np.float32)
    pb = np.array([[10,10,50,50],[60,60,100,100]],dtype=np.float32)
    ps = np.array([0.9,0.8],dtype=np.float32)
    e.update(pb,ps,gt)
    assert e.compute()["mAP@0.5"] > 0.9
    print("✅")

def test_boxes():
    print("boxes...", end=" ")
    from src.utils.boxes import xyxy2cxcywh, cxcywh2xyxy
    b = torch.tensor([[10.,10.,50.,50.]])
    assert torch.allclose(cxcywh2xyxy(xyxy2cxcywh(b)), b, atol=1e-4)
    print("✅")

if __name__ == "__main__":
    print("\n=== PHD Sanity Tests ===\n")
    test_backbone(); test_neck(); test_model_train()
    test_model_infer(); test_loss(); test_metrics(); test_boxes()
    print("\n🎉 All passed!\n")