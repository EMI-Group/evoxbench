package evoxgroup;
import evoxgroup.EvoXException;
import evoxgroup.Benchmark;
import evoxgroup.BenchmarkFactory;
import evoxgroup.Config;
import evoxgroup.Args;

public class IN1KMOP extends BenchmarkFactory {
    IN1KMOP(String address, int port) {
        super(address, port);
    }

    Benchmark createBenchmark(int problemId) throws EvoXException {
        Config config;
        Args args = new Args();

        switch (problemId) {
            case 1:
                args.objs = "err&params";
                args.normalized_objectives = false;
                config = new Config("resnet", args);
                break;
            case 2:
                Args args = new Args();
                args.objs = "err&flops";
                args.normalized_objectives = false;
                config = new Config("resnet", args);
                break;
            case 3:
                Args args = new Args();
                args.objs = "err&params&flops";
                args.normalized_objectives = false;
                config = new Config("resnet", args);
                break;
            case 7:
                Args args = new Args();
                args.objs = "err&params";
                args.normalized_objectives = false;
                config = new Config("mnv3", args);
                break;
            case 8:
                Args args = new Args();
                args.objs = "err&params&flops";
                args.normalized_objectives = false;
                config = new Config("mnv3", args);
                break;
            case 9:
                Args args = new Args();
                args.objs = "err&params&flops&latency";
                args.normalized_objectives = false;
                config = new Config("mnv3", args);
                break;
            default:
                throw new EvoXException("No such problem id");
        }
        Benchmark bm = new Benchmark();
        bm.init(address, port, config);

        return bm;
    }
}