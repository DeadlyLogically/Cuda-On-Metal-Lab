#!/usr/bin/env python3
import argparse
import ctypes as c
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import signal
import socket
import sys

U32, U64, P = c.c_uint32, c.c_uint64, c.c_void_p
MAX_BYTES = 8 * 1024 * 1024
MAX_SESSION_BYTES = 64 * 1024 * 1024
MAX_ARGS = 32
ERROR_INVALID_VALUE, ERROR_INVALID_HANDLE, ERROR_NOT_SUPPORTED = 1, 400, 801
PTX_NAME = r'[a-zA-Z_$][a-zA-Z0-9_$]*'
PTR_PARAM = re.compile(r'\.param\s+\.(?:u64|b64)\s+\.ptr(?:\s+\.global)?(?:\s+\.align\s+[1-9]\d*)?\s+' + PTX_NAME)

class DriverError(Exception):
    def __init__(self, status, operation):
        super().__init__(operation)
        self.status = status

def require(condition, message, code=ERROR_INVALID_VALUE):
    if not condition:
        raise DriverError(code, message)

def number(text, maximum=(1 << 64) - 1):
    require(re.fullmatch(r'[0-9]{1,20}', text) is not None, 'invalid integer')
    value = int(text)
    require(value <= maximum, 'integer out of range')
    return value

def bytes_field(text):
    require(0 < len(text) <= 2 * MAX_BYTES and len(text) % 2 == 0, 'invalid hex length')
    try:
        return bytes.fromhex(text)
    except ValueError:
        raise DriverError(ERROR_INVALID_VALUE, 'invalid hex data') from None

class Driver:
    def __init__(self, path):
        self.lib = c.CDLL(str(path))
        types = {
            'cuInit': [U32], 'cuDeviceGetCount': [c.POINTER(c.c_int)],
            'cuCtxCreate': [c.POINTER(P), U32, c.c_int], 'cuCtxDestroy': [P],
            'cuDeviceGetAttribute': [c.POINTER(c.c_int), c.c_int, c.c_int],
            'cudaDeviceGetAttribute': [c.POINTER(c.c_int), c.c_int, c.c_int],
            'cuDeviceGetUuid': [P, c.c_int], 'cuDeviceGetName': [P, c.c_int, c.c_int],
            'cuDeviceTotalMem': [c.POINTER(c.c_size_t), c.c_int],
            'cuCtxSynchronize': [], 'cuModuleLoadData': [c.POINTER(P), P],
            'cuModuleGetFunction': [c.POINTER(P), P, c.c_char_p], 'cuModuleUnload': [P],
            'cuMemAlloc': [c.POINTER(U64), c.c_size_t], 'cuMemFree': [U64],
            'cuMemcpyHtoD': [U64, P, c.c_size_t], 'cuMemcpyDtoH': [P, U64, c.c_size_t],
            'cuLaunchKernel': [P, U32, U32, U32, U32, U32, U32, U32, P, c.POINTER(P), P],
        }
        for name, args in types.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, c.c_int

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise DriverError(status, name)

@dataclass
class Allocation:
    host: int
    size: int

class Session:
    def __init__(self, driver):
        self.driver = driver
        self.ctx = P()
        self.allocations = {}
        self.modules = {}
        self.functions = {}
        self.next_address = 0x100000000
        self.next_handle = 1

    def handle(self, table, obj):
        value = self.next_handle
        self.next_handle += 1
        table[value] = obj
        return str(value)

    def host_ptr(self, guest, size):
        for base, block in self.allocations.items():
            offset = guest - base
            if 0 <= offset < block.size and size <= block.size - offset:
                return block.host + offset
        raise DriverError(ERROR_INVALID_VALUE, 'unknown or out-of-bounds guest address')

    def device_attribute(self, attribute):
        value = c.c_int()
        try:
            self.driver.call('cuDeviceGetAttribute', c.byref(value), attribute, 0)
        except DriverError as driver_error:


            if driver_error.status != ERROR_INVALID_VALUE or attribute not in (39, 81):
                raise
            try:
                self.driver.call('cudaDeviceGetAttribute', c.byref(value), attribute, 0)
            except DriverError as runtime_error:
                if runtime_error.status != ERROR_INVALID_VALUE or attribute != 81:
                    raise

                self.driver.call('cuDeviceGetAttribute', c.byref(value), 8, 0)
        return str(value.value)

    def request(self, op, args):
        arities = {'INIT': 0, 'ALLOC': 1, 'H2D': 2, 'D2H': 2, 'FREE': 1, 'SYNC': 0,
                   'LOAD': 1, 'GET': 2, 'UNLOAD': 1, 'ATTR': 1, 'UUID': 0, 'NAME': 0, 'TOTAL': 0}
        require(op in arities or op == 'LAUNCH', 'unknown operation', ERROR_NOT_SUPPORTED)
        if op in arities:
            require(len(args) == arities[op], 'argument count')
        call = self.driver.call
        if op == 'INIT':
            if not self.ctx.value:
                call('cuInit', 0)
                count = c.c_int()
                call('cuDeviceGetCount', c.byref(count))
                require(count.value > 0, 'no GPU', 100)
                call('cuCtxCreate', c.byref(self.ctx), 0, 0)
            return ''
        require(self.ctx.value, 'init required', 3)
        if op == 'NAME':
            name = c.create_string_buffer(256)
            call('cuDeviceGetName', name, len(name), 0)
            return name.value.hex()
        if op == 'TOTAL':
            size = c.c_size_t()
            call('cuDeviceTotalMem', c.byref(size), 0)
            return str(size.value)
        if op == 'ATTR':
            return self.device_attribute(number(args[0], (1 << 31) - 1))
        if op == 'UUID':
            value = c.create_string_buffer(16)
            call('cuDeviceGetUuid', value, 0)
            return value.raw.hex()
        if op == 'ALLOC':
            n = number(args[0], MAX_BYTES)
            require(n > 0, 'empty allocation')
            require(n + sum(b.size for b in self.allocations.values()) <= MAX_SESSION_BYTES, 'memory quota', 2)
            host = U64()
            call('cuMemAlloc', c.byref(host), n)
            guest = self.next_address
            self.next_address += ((n + 4095) // 4096 + 1) * 4096
            self.allocations[guest] = Allocation(host.value, n)
            return str(guest)
        if op == 'FREE':
            guest = number(args[0])
            require(guest in self.allocations, 'unknown allocation')
            call('cuMemFree', self.allocations[guest].host)
            del self.allocations[guest]
            return ''
        if op == 'H2D':
            raw = bytes_field(args[1])
            ptr = self.host_ptr(number(args[0]), len(raw))
            call('cuMemcpyHtoD', ptr, c.create_string_buffer(raw, len(raw)), len(raw))
            return ''
        if op == 'D2H':
            size = number(args[1], MAX_BYTES)
            require(size > 0, 'empty read')
            ptr = self.host_ptr(number(args[0]), size)
            raw = c.create_string_buffer(size)
            call('cuMemcpyDtoH', raw, ptr, size)
            return raw.raw.hex()
        if op == 'SYNC':
            call('cuCtxSynchronize')
            return ''
        if op == 'LOAD':
            raw = bytes_field(args[0]).removesuffix(b'\0')
            require(b'\0' not in raw, 'embedded NUL')
            try:
                source = raw.decode('ascii')
            except UnicodeDecodeError:
                raise DriverError(ERROR_INVALID_VALUE, 'expected ASCII PTX') from None
            source = re.sub(r'/\*.*?\*/|//[^\n]*', '', source, flags=re.S)
            entries = {}
            for name, declarations in re.findall(r'\.entry\s+('+PTX_NAME+r')\s*\(([^()]*)\)', source):
                params = [p.strip() for p in declarations.split(',')] if declarations.strip() else []
                entries[name] = (len(params) if len(params) <= MAX_ARGS and
                                 all(PTR_PARAM.fullmatch(p) for p in params) else None)
            require(entries, 'no PTX entry point', ERROR_NOT_SUPPORTED)
            handle = P()
            call('cuModuleLoadData', c.byref(handle), c.create_string_buffer(raw))
            return self.handle(self.modules, (handle, entries))
        if op == 'GET':
            mod = number(args[0])
            require(mod in self.modules, 'unknown module', ERROR_INVALID_HANDLE)
            handle, entries = self.modules[mod]
            name = args[1]
            require(name in entries, 'unknown function', 500)
            require(entries[name] is not None, 'only explicit pointer parameters supported', ERROR_NOT_SUPPORTED)
            fn = P()
            call('cuModuleGetFunction', c.byref(fn), handle, name.encode('ascii'))
            return self.handle(self.functions, (fn, mod, entries[name]))
        if op == 'UNLOAD':
            mod = number(args[0])
            require(mod in self.modules, 'unknown module', ERROR_INVALID_HANDLE)
            call('cuCtxSynchronize')
            call('cuModuleUnload', self.modules[mod][0])
            del self.modules[mod]
            self.functions = {key: val for key, val in self.functions.items() if val[1] != mod}
            return ''
        require(len(args) >= 4, 'short launch')
        fn = number(args[0])
        require(fn in self.functions, 'unknown function', ERROR_INVALID_HANDLE)
        handle, _, expected = self.functions[fn]
        grid, block, count = number(args[1], 65535), number(args[2], 1024), number(args[3], MAX_ARGS)
        require(grid > 0 and block > 0 and count == len(args) - 4 and count == expected, 'invalid launch shape or signature')
        values = []
        for arg in args[4:]:
            require(arg.startswith('p:'), 'pointer arguments required', ERROR_NOT_SUPPORTED)
            values.append(U64(self.host_ptr(number(arg[2:]), 1)))
        ptrs = (P * count)(*(c.addressof(value) for value in values))
        call('cuLaunchKernel', handle, grid, 1, 1, block, 1, 1, 0, None, ptrs, None)
        return ''

    def close(self):
        if self.ctx.value:
            for name, args in ([('cuCtxSynchronize', ())] +
                               [('cuModuleUnload', (h,)) for h, _ in self.modules.values()] +
                               [('cuMemFree', (a.host,)) for a in self.allocations.values()] +
                               [('cuCtxDestroy', (self.ctx,))]):
                try:
                    self.driver.call(name, *args)
                except DriverError as e:
                    logging.error('cleanup %s failed with CUDA status %d', name, e.status)

def serve_client(conn, driver, session_id):
    session = Session(driver)
    try:
        with conn.makefile('rb') as reader:
            while True:
                line = reader.readline(2 * MAX_BYTES + 1024)
                if not line:
                    break
                if not line.endswith(b'\n'):
                    conn.sendall(b'ERR 1\n')
                    break
                op = '?'
                try:
                    fields = line.decode('ascii').split()
                    require(fields, 'empty request')
                    op, *args = fields
                    result = session.request(op, args)
                    reply = 'OK' + (' ' + result if result else '')
                    error = ''
                except (ValueError, UnicodeError):
                    reply, error = 'ERR 1', 'invalid request encoding'
                except DriverError as e:
                    reply, error = f'ERR {e.status}', str(e)
                print(json.dumps({'session': session_id, 'op': op, 'request_sha256': hashlib.sha256(line).hexdigest(),
                                  'status': reply.split()[0:2] if reply.startswith('ERR') else 'OK',
                                  'error': error}), flush=True)
                conn.sendall((reply + '\n').encode('ascii'))
    finally:
        session.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--socket', required=True, help='new private UNIX socket (use a short path on macOS)')
    parser.add_argument('--max-sessions', type=int, default=0)
    args = parser.parse_args()
    os.environ.update(CUMETAL_TRACE_GPU='1', CUMETAL_DRIVER_JIT='runtime-msl', CUMETAL_PTX_BACKEND='cumetal-ir',
                      CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS='0', CUMETAL_ENABLE_LLMC_CPU_EMULATION='0',
                      CUMETAL_DISABLE_LLMC_EMULATION='1', CUMETAL_DEBUG_ERRORS='1')
    driver = Driver(args.library.resolve(strict=True))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(args.socket)
        try:
            os.chmod(args.socket, 0o600)
            server.listen(4)
            print(json.dumps({'ready': args.socket, 'backend': str(args.library.resolve())}), flush=True)
            session_count = 0
            while not args.max_sessions or session_count < args.max_sessions:
                conn, _ = server.accept()
                session_count += 1
                with conn:
                    try:
                        serve_client(conn, driver, session_count)
                    except ConnectionError as e:
                        logging.info('client connection ended: %s', e)
        finally:
            os.unlink(args.socket)

if __name__ == '__main__':
    main()
