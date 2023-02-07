package evoxgroup;

import evoxgroup.Benchmark;

abstract class BenchmarkFactory {
    String address;
    int port;

    BenchmarkFactory(String address, int port) {
        this.address = address;
        this.port = port;
    }

    abstract Benchmark createBenchmark(int problemId) throws EvoXException;
}