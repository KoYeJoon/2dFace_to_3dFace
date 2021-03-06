from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--test_img_path', type=str, default='./data_input/', help='path of single test image.')
        parser.add_argument('--test_upscale', type=int, default=1, help='upscale single test image.')
        parser.add_argument('--results_dir', type=str, default='./fuse_deep3d/data/input', help='saves results here.')
        parser.add_argument('--save_as_dir', type=str, default='', help='save results in different dir.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--pretrain_model_path', type=str, default='', help='load pretrain model path if specified')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # rewrite devalue values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        # pyqt_version_
        parser.add_argument('--pyqt_ver', type=str, default='terminal', help='what your input type image/dir')
        # parser for image/video
        parser.add_argument('--type', type=str, default='dir', help='what your input type image/dir')
        # parser for deep3d code
        parser.add_argument('--use_pb', type=int, default=1, help='validation data folder')
        parser.add_argument('--objface_results_dir', type=str, default='./data_output', help='saves 3d face results here.')
        self.isTrain = False
        return parser
