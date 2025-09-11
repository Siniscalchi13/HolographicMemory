import asyncio
import os
import pty
import shlex
import signal
import fcntl
import termios
import struct
from typing import Optional


class ShellSession:
    """
    Manages a PTY-backed shell process and bridges I/O.
    """

    def __init__(self, shell: Optional[str] = None):
        self.shell = shell or os.environ.get("SHELL", "/bin/bash")
        self.master_fd: Optional[int] = None
        self.pid: Optional[int] = None

    def spawn(self, cols: int = 120, rows: int = 32) -> None:
        self.pid, self.master_fd = pty.fork()
        if self.pid == 0:
            # Child: exec shell
            os.execvp(self.shell, [self.shell])
        else:
            # Parent: set window size
            self.set_winsize(rows, cols)

    def set_winsize(self, rows: int, cols: int) -> None:
        if self.master_fd is None:
            return
        # TIOCSWINSZ = winsize struct: rows, cols, xpix, ypix
        winsize = struct.pack("HHHH", rows, cols, 0, 0)
        fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, winsize)

    async def read_loop(self, callback):
        """Continuously read from PTY and push to callback"""
        if self.master_fd is None:
            return
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, os.fdopen(self.master_fd, 'rb', buffering=0))

        try:
            while True:
                data = await reader.read(1024)
                if not data:
                    break
                await callback(data)
        except (asyncio.CancelledError, Exception):
            return

    async def write(self, data: bytes) -> None:
        if self.master_fd is None:
            return
        os.write(self.master_fd, data)

    def terminate(self) -> None:
        if self.pid:
            try:
                os.kill(self.pid, signal.SIGHUP)
            except ProcessLookupError:
                pass
        if self.master_fd:
            try:
                os.close(self.master_fd)
            except OSError:
                pass

