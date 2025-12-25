from bench2driveMMCV.utils import get_git_hash
from bench2driveMMCV import __version__

def collect_env():
    """Collect the information of the running environments."""
    env_info = {}
    env_info['bench2driveMMCV'] = __version__
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')
