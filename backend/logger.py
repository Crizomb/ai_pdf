import gradio as gr
import sys


class Logger:
    """
    Logger class to redirect the output to a file.
    will be used to the log textbox in the frontend.

    Adapted from  : https://github.com/gradio-app/gradio/issues/2362#issuecomment-1424446778
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


def read_logs():
    sys.stdout.flush()
    with open("../temp_file/output.log", "r") as f:
        return f.read()


