from typing import Mapping
import re
import signal
from contextlib import contextmanager
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import io
from typing import Any

class PythonREPL:
    """A tool for running python code in a REPL."""

    name = "PythonREPL"
    # This PythonREPL is not used by the environment; It is THE ENVIRONMENT.
    signature = "NOT_USED"
    description = "NOT_USED"

    def __init__(
        self,
        user_ns: Mapping[str, Any],
        # max_observation_length: int = 1024,
        timeout: int = 30,
    ) -> None:
        super().__init__()
        self.user_ns = user_ns
        # self.max_observation_length = max_observation_length
        self.timeout = timeout
        self.reset()

    @contextmanager
    def time_limit(self, seconds):
        def signal_handler(signum, frame):
            raise TimeoutError(f"Timed out after {seconds} seconds. Consider change your code to reduce the running time.")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)  # Disable the alarm

    def reset(self) -> None:
        self.shell = InteractiveShell(
            # NOTE: shallow copy is needed to avoid
            # shell modifying the original user_ns dict
            user_ns=dict(self.user_ns),
            colors="NoColor",
        )

    def __call__(self, query: str) -> str:
        """Use the tool and return observation"""
        with self.time_limit(self.timeout):
            # NOTE: The timeout error will be caught by the InteractiveShell

            # Capture all output
            with io.capture_output() as captured:
                _ = self.shell.run_cell(query, store_history=True)
            output = captured.stdout

            if output == "":
                output = "[Executed Successfully with No Output]"

            # replace potentially sensitive filepath
            # e.g., File /mint/mint/tools/python_tool.py:30, in PythonREPL.time_limit.<locals>.signal_handler(signum, frame)
            # with File <filepath>:30, in PythonREPL.time_limit.<locals>.signal_handler(signum, frame)
            # use re
            output = re.sub(
                # r"File (/mint/)mint/tools/python_tool.py:(\d+)",
                r"File (.*).py:(\d+)",
                r"File <hidden_filepath>:\2",
                output,
            )

            # if len(output) > self.max_observation_length:
            #     # make sure the beginning and the end of the output are not truncated
            #     output = (
            #         output[: self.max_observation_length // 2]
            #         + "\n[...truncated due to length...]\n"
            #         + output[-self.max_observation_length // 2 :]
            #     )

        return output

    def __del__(self):
        if self.shell is None:
            return
        self.shell.reset()
        self.shell.cleanup()
        self.shell = None
