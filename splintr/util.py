from datetime import datetime
verbose = False

def vprint(msg):
    if verbose:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'[{timestamp}]\t{msg}')