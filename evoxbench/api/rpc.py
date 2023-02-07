from glob import glob
import json
from socketserver import StreamRequestHandler, ThreadingTCPServer
from io import TextIOWrapper
from collections import OrderedDict
import time
import numpy as np
import uuid
import argparse
import traceback
from evoxbench import benchmarks
from evoxbench.modules import Benchmark

DEBUG_FLAG = False
# default is 10s
# READ_TIMEOUT = 5*60

"""
EvoX Simple Protocol

Create Object:
    {
        'operation' = 'create',
        'config': a dict, indicating which benchmark and what config to use,
    }
return:
    {
        'status': 'ok'
    }

Query result:
    {
        'operation' = 'query',
        'encoding': a vector (list)
    }
return:
    {
        'status': 'ok',
        'result': a vector, containing f1, f2, f3 ...
    }

"""


def handle_numpy(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


def new_benchmark_obj(config):
    name = config['name'].lower()
    args = config.get('args', {})

    if name == 'nb101':
        obj = benchmarks.NASBench101Benchmark(**args)
    elif name == 'nb201':
        obj = benchmarks.NASBench201Benchmark(**args)
    elif name == 'darts':
        obj = benchmarks.DARTSBenchmark(**args)
    elif name == 'nats':
        obj = benchmarks.NATSBenchmark(**args)
    elif name == 'mnv3':
        obj = benchmarks.MobileNetV3Benchmark(**args)
    elif name == 'resnet':
        obj = benchmarks.ResNet50DBenchmark(**args)
    elif name == 'transformer':
        obj = benchmarks.TransformerBenchmark(**args)
    return obj


def reject_message(reason):
    return json.dumps({
        'status': 'err',
        'reason': reason
    }).encode('utf-8')


def handle_request(request, benchmark_obj):
    op = request['operation']
    try:
        if op == 'query':
            encoding = request['encoding']
            true_eval = request['true_eval']
            # encoding = [np.array(x) for x in encoding]
            encoding = np.array(encoding)
            if len(encoding.shape) == 1:
                # query for only one point
                encoding = encoding[np.newaxis, :]
            assert (isinstance(benchmark_obj, Benchmark))
            result = benchmark_obj.evaluate(encoding, true_eval=true_eval)
            return json.dumps({
                'status': 'ok',
                'result': result
            }, default=handle_numpy).encode('utf-8')
        elif op == 'sample':
            assert (isinstance(benchmark_obj, Benchmark))

            n_samples = request['n_samples']

            result = benchmark_obj.search_space.encode(
                benchmark_obj.search_space.sample(n_samples))

            return json.dumps({
                'status': 'ok',
                'result': result
            }, default=handle_numpy).encode('utf-8')
        elif op == 'pareto_front':
            assert (isinstance(benchmark_obj, Benchmark))

            pf = benchmark_obj.pareto_front
            if pf is None:
                n_samples = request.get('n_samples', 1)
                # had to do a sample, because the shape is not known
                sample = benchmark_obj.search_space.encode(
                    benchmark_obj.search_space.sample(n_samples))
                pf = np.zeros_like(sample)
            else:
                n_samples = request.get('n_samples', -1)
                if benchmark_obj.normalized_objectives:
                    pf = benchmark_obj.normalize(pf)
                # a valid n_samples, return only n_samples points on pareto front
                # otherwise, return all points
                if n_samples > 0:
                    sampled_indices = np.random.choice(pf.shape[0], n_samples)
                    pf = pf[sampled_indices]

                indices = np.lexsort(pf.T)
                pf = pf[indices]

            return json.dumps({
                'status': 'ok',
                'result': pf
            }, default=handle_numpy).encode('utf-8')
        elif op == 'calc_perf_indicator':
            assert (isinstance(benchmark_obj, Benchmark))
            inputs = request['inputs']
            inputs = np.array(inputs)
            indicator = request['indicator']
            result = benchmark_obj.calc_perf_indicator(inputs, indicator)
            return json.dumps({
                'status': 'ok',
                'result': result
            }, default=handle_numpy).encode('utf-8')
        elif op == 'call':
            if not DEBUG_FLAG:
                return json.dumps({
                    'status': 'err'
                })

            # call any method remotely
            # Warning: do NOT use is on the internet
            member = request['member']
            method_name = request['method']
            args = request['args']
            assert (isinstance(benchmark_obj, Benchmark))
            member = getattr(benchmark_obj, member)
            method = getattr(member, method_name)
            result = method(**args)

            return json.dumps({
                'status': 'ok',
                'result': result
            }, default=handle_numpy).encode('utf-8')
        elif op == 'settings':
            defaultsettings = {
                'n_var': benchmark_obj.search_space.n_var,
                'lb': benchmark_obj.search_space.lb,
                'ub': benchmark_obj.search_space.ub,
                'n_obj': benchmark_obj.evaluator.objs.count('&') + 1
            }
            return json.dumps({
                'status': 'ok',
                'result': defaultsettings
            }, default=handle_numpy).encode('utf-8')
        else:
            return reject_message('No such operation')
    except Exception as e:
        print(e)
        traceback.print_exc()
        return reject_message('Failed to create new objects')


class EvoXSimpleRPC(StreamRequestHandler):
    def _return_err(self):
        msg = json.dumps({
            'status': 'err'
        })
        self.wfile.write(msg.encode('utf-8'))
        self.wfile.write(b'\n')
        self.wfile.flush()

    def handle(self):
        try:
            benchmark_obj = None
            while True:
                req_string = self.rfile.readline().decode('utf-8')
                try:
                    request = json.loads(req_string)
                except json.decoder.JSONDecodeError:
                    print(f'Json decode error, the string is "{req_string}"')
                    # not a json
                    self._return_err()
                    return
                # special cases
                if 'operation' not in request:
                    self._return_err()
                    return
                elif request['operation'] == 'delete':
                    ## gracefully exit
                    print("Exited")
                    return
                elif request['operation'] == 'create':
                    ## create
                    config = request['config']
                    benchmark_obj = new_benchmark_obj(config)
                    body = json.dumps({
                        'status': 'ok'
                    }).encode('utf-8')
                else:
                    body = handle_request(request, benchmark_obj)

                self.wfile.write(body)
                self.wfile.write(b'\n')
                self.wfile.flush()
        except Exception as e:
            print(e)
            traceback.print_exc()
            print("Force exit")


def start_server(listen_addr=('127.0.0.1', 9876)):
    http_server = ThreadingTCPServer(listen_addr, EvoXSimpleRPC)
    http_server.serve_forever()


def main():
    global DEBUG_FLAG
    # evoxbenchrpc 127.0.0.1:9876
    parser = argparse.ArgumentParser(description='ExoXBench rpc server')
    parser.add_argument('-l', '--address', default='127.0.0.1:9876')
    parser.add_argument('--debug', help='Enable debug', action='store_true')
    args = parser.parse_args()

    addr, port = args.address.split(':')
    port = int(port)
    DEBUG_FLAG = args.debug

    print(f"Starting rpc server at {addr}, port {port}")

    start_server((addr, port))
