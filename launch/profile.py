#!/usr/bin/env python3
import argparse
import array
from contextlib import ExitStack
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import sys
import time

ROOT = Path(__file__).resolve().parents[1] / "sass"
sys.path.insert(0, str(ROOT / "mac"))
from run_gpu import Driver, U64, I32, PTR

def profile(gpu, fn, n, batch, trials):
    block = 256
    grid = (n + block - 1) // block
    capacity = grid * block + 16
    left = array.array('f', (((i*17+3)%251-125)/8 for i in range(n)))
    addend = array.array('f', (((i*29+7)%239-119)/16 for i in range(n)))
    guard = array.array('f', [-12345.25]) * (capacity-n)
    start = left + guard
    delta = addend + guard
    expected = array.array('f', (left[i]+batch*addend[i] for i in range(n))) + guard
    fresh = array.array('f', [-12345.25]) * capacity
    ctypes_buf = C.c_float * capacity
    aa, bb, ee = (ctypes_buf.from_buffer(storage) for storage in (start, delta, fresh))
    size = C.sizeof(aa)
    with ExitStack() as stack:
        pointers=[]
        for host in (aa,bb,ee):
            ptr=U64()
            gpu.call('cuMemAlloc', C.byref(ptr), size)
            stack.callback(gpu.call, 'cuMemFree', ptr)
            pointers.append(ptr)
            gpu.call('cuMemcpyHtoD', ptr, host, size)
        count=I32(n)
        argtype=PTR*4
        argv0=argtype(*(C.addressof(v) for v in (pointers[0],pointers[1],pointers[2],count)))
        argv1=argtype(*(C.addressof(v) for v in (pointers[2],pointers[1],pointers[0],count)))
        args=(argv0,argv1)
        collected={mode:[] for mode in ('serialized','queued')}
        def one(mode):
            gpu.call('cuMemcpyHtoD',pointers[0],aa,size)
            gpu.call('cuMemcpyHtoD',pointers[2],ee,size)
            gpu.call('cuCtxSynchronize')
            launches=0
            waits=0
            t0=time.perf_counter_ns()
            for j in range(batch):
                begin=time.perf_counter_ns()
                gpu.call('cuLaunchKernel',fn,grid,1,1,block,1,1,0,None,args[j%2],None)
                enqueued=time.perf_counter_ns()
                launches+=enqueued-begin
                if mode=='serialized' or j==batch-1:
                    gpu.call('cuCtxSynchronize')
                    waits+=time.perf_counter_ns()-enqueued
            wall=time.perf_counter_ns()-t0
            result=ctypes_buf()
            gpu.call('cuMemcpyDtoH',result,pointers[2 if batch%2 else 0],size)
            assert bytes(result)==expected.tobytes(),(n,batch,mode)
            return {'wall_us_per_op':round(wall/(1000*batch),3),
                    'launch_us_per_op':round(launches/(1000*batch),3),
                    'wait_us_per_op':round(waits/(1000*batch),3)}
        for mode in collected: one(mode)
        for t in range(trials):
            modes=['serialized','queued']
            if t%2: modes.reverse()
            for mode in modes:
                collected[mode].append(one(mode))
        return {'n':n,'batch':batch,'digest':hashlib.sha256(expected.tobytes()).hexdigest(),
                'raw':collected,'median':{m:{k:median([z[k] for z in rows]) for k in rows[0]} for m,rows in collected.items()}}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--library', type=Path, required=True)
    ap.add_argument('--counts',type=int,nargs='+',default=[32,50000,524288])
    ap.add_argument('--batches',type=int,nargs='+',default=[1,16])
    ap.add_argument('--trials',type=int,default=9)
    ap.add_argument('--trace-gpu',action='store_true')
    args=ap.parse_args()
    os.environ.update(CUMETAL_DRIVER_JIT='runtime-msl',CUMETAL_PTX_BACKEND='cumetal-ir',
            CUMETAL_SASS_LIFTER=str(ROOT/'lifter/lift.py'),CUMETAL_SASS_NVDISASM=str(ROOT/'mac/nvdisasm-docker'),
            CUMETAL_SASS_PYTHON=sys.executable,CUMETAL_TRACE_GPU='1' if args.trace_gpu else '0',
            CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS='0',CUMETAL_ENABLE_LLMC_CPU_EMULATION='0',CUMETAL_DISABLE_LLMC_EMULATION='1')
    binfile=ROOT/'fixtures/build/vectorAdd.sm75.cubin'
    gpu=Driver(args.library)
    gpu.call('cuInit',0)
    with ExitStack() as stack:
        ctx=PTR();gpu.call('cuCtxCreate',C.byref(ctx),0,0);stack.callback(gpu.call,'cuCtxDestroy',ctx)
        dev=C.create_string_buffer(256);gpu.call('cuDeviceGetName',dev,len(dev),0)
        mod=PTR();gpu.call('cuModuleLoad',C.byref(mod),os.fsencode(binfile));stack.callback(gpu.call,'cuModuleUnload',mod)
        fn=PTR();gpu.call('cuModuleGetFunction',C.byref(fn),mod,b'vectorAdd')
        report={'hardware':dev.value.decode(),'cubin_sha256':hashlib.sha256(binfile.read_bytes()).hexdigest(),
                'lib_sha256':hashlib.sha256(args.library.read_bytes()).hexdigest(),'trace_gpu':args.trace_gpu,'cases':[]}
        for n in args.counts:
            for batch in args.batches:
                row=profile(gpu,fn,n,batch,args.trials);report['cases'].append(row)
                print(n,batch,row['median'],flush=True)
        args.out.write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
