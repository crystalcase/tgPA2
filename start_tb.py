from tensorboard import program
from utils import *

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', get_tb_dir()])
url = tb.launch()
