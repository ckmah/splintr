import math
from datetime import datetime
verbose = False

def vprint(msg):
    if verbose:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{timestamp}]\t{msg}')
        
def calc_conv_pad(input_size, output_size, kernel_size, stride):
    '''
    Calculate appropriate padding to guarantee output size.
    '''
    return math.ceil((output_size * stride - input_size + kernel_size - stride) / 2)

def calc_conv_stride(input_size, output_size, kernel_size, pad_size):
    return math.floor((input_size - kernel_size + 2 * pad_size) / (output_size - 1))