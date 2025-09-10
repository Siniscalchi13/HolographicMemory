#!/usr/bin/env python3
"""
Find a free port starting from 8080
"""
import socket

def find_free_port(start=8080, max_attempts=10):
    for port in range(start, start + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return start

if __name__ == "__main__":
    print(find_free_port())
