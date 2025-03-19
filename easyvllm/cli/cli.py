import fire
from easyvllm.cli.decode import decode, decode_multi_task, test

class Cli:
    def __init__(self):
        self.decode = decode
        self.decode.multask = decode_multi_task
        self.decode.test = test


def main():
    fire.Fire(Cli)

