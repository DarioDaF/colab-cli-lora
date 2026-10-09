# /// script
# dependencies = [
#   "python-dotenv>=1.2.4",
# ]
# ///

from contextlib import contextmanager
import subprocess as sp
import random
import dotenv
from pathlib import Path

THIS_FOLDER = Path(__file__).resolve().parent

class ColabSession:
    #COLAB_CMD = [ os.environ['HOME'] + '/.local/bin/uv', 'tool', 'run', '--from', 'google-colab-cli', 'colab' ]
    COLAB_CMD = [ 'colab' ]
    VM_TYPE = {
        'cpu': [],
        'tpu': [ '--tpu', 'v5e1' ],
        'gpu': [ '--gpu', 'T4' ],
    }

    def __init__(self, vm_type: str = 'cpu', sid: str|None = None) -> None:
        self.vm_type = vm_type
        self.sid = sid if sid is not None else ''.join(random.choices('abcdefghijklmnopqrtsuvwxyz0123456789', k=8))
        self._keep = False
    def sp_run(self, args: list[str], **kwargs):
        return sp.run([ *ColabSession.COLAB_CMD, args[0], '--session', self.sid, *args[1:] ], check=True, **kwargs)
    def start(self):
        self.sp_run([ 'new', *ColabSession.VM_TYPE[self.vm_type] ])
    def keep(self):
        self._keep = True
    def stop(self):
        if self._keep:
            print(f'!!! WARNING: Colab session "{self.sid}" was intentionally left open')
        else:
            self.sp_run([ 'stop' ])
    @contextmanager
    @staticmethod
    def with_session(vm_type: str = 'cpu', sid: str|None = None, start: bool = True, keep: bool = False):
        s = ColabSession(vm_type, sid)
        if start:
            s.start()
        if keep:
            s.keep()
        try:
            yield s
        finally:
            s.stop()

def assert_not_none[T](v: T|None) -> T:
    assert(v is not None)
    return v

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sid', help='Colab session id to connect to existing session')
    parser.add_argument('--keep', help='Keep session open after execution', action='store_true')
    parser.add_argument('--start', help='Force start of a runtime even when sid is provided', action='store_true')
    parser.add_argument('--timeout', help='Execution timeout in seconds', default=f'{1*60*60}')
    args = parser.parse_args()
    arg_sid: str|None = args.sid
    arg_keep: bool = args.keep
    arg_start: bool = args.start
    arg_timeout: str = args.timeout

    print('### OPENING SESSION')
    with ColabSession.with_session(sid=arg_sid, start=(arg_start or (arg_sid is None)), keep=arg_keep, vm_type='gpu') as s:
        envDict = dotenv.dotenv_values()
        # @TODO: @WARN: ssh proxy with colab will create a runtime if not already present, this can mess up 'vm_type' !!!
        print('### SYNCING FILES')
        sp.run([ 'colab-rsync', s.sid, './code/', 'remote:/content/code' ], check=True)
        envParams = []
        for k, v in envDict.items():
            assert('=' not in k)
            envParams.append('--env')
            envParams.append(f'{k}={v}')
        print('### RUNNING CODE')
        s.sp_run([ 'exec', '--timeout', arg_timeout, *envParams ], input='!cd /content/code && bash ./run.sh\n', encoding='utf-8')

if __name__ == '__main__':
    main()
