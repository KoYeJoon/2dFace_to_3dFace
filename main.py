import sys
import os
sys.path.append("./SR/src")
sys.path.append("./fuse_deep3d")
# sys.path.append("./fuse_deep3d/src_3d")
from fuse_deep3d import reconstruction3d
from SR.src import test_enhance_single_unalign
from fuse_deep3d import reconstruction3d
from SR.src.options.test_options import TestOptions

if __name__ == '__main__':
    opt = test_enhance_single_unalign.sr_demo()
    reconstruction3d.demo(opt)
