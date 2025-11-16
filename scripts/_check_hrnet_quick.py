print("=== HRNet quick check ===")
import numpy
print("numpy:", numpy.__version__)
try:
    from mmpose.models.backbones import HRNet as MMPoseHRNet
    print("OK: MMPoseHRNet imported:", MMPoseHRNet)
except Exception as e:
    import traceback
    print("IMPORT_ERROR:", repr(e))
    traceback.print_exc()
