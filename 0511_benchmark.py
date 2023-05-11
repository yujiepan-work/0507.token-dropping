import itertools
import subprocess
from argparse import Namespace
from dataclasses import dataclass
from typing import List
import pandas as pd
from pathlib import Path

@dataclass
class BenchmarkResult:
    stdout: str = ''
    stderr: str = ''
    avg_latency: float = 0.
    throughput: float = 0.


def run_benchmark(cmd):
    with subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        p.wait()
        stdout = p.stdout.read().decode()
        stderr = p.stderr.read().decode()
        print(stdout)
        print(stderr)
        return BenchmarkResult(
            stdout=stdout,
            stderr=stderr,
            throughput=list(filter(lambda x: x.strip(), stdout.split('\n')))[-1].split()[-2],
            avg_latency=list(filter(lambda x: x.strip(), stdout.split('\n')))[-4].split()[-2],
        )


results = []
for model in Path('./0511_export_onnx/').glob('**/model.onnx'):
    print('***' * 20)
    model = model.absolute()
    print(model)

    result = run_benchmark(cmd=f'benchmark_app -m {model} -niter 3000 -hint latency')
    results.append(
        dict(
            model=model,
            latency=result.avg_latency
        )
    )

    df = pd.DataFrame(results)
    df.to_csv('./0511_export_onnx/fp-spr.csv')
