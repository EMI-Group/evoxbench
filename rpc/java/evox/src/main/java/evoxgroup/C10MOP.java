package evoxgroup;
import evoxgroup.EvoXException;
import evoxgroup.Benchmark;
import evoxgroup.BenchmarkFactory;
import evoxgroup.Config;
import evoxgroup.Args;

public class C10MOP extends BenchmarkFactory {
    C10MOP(String address, int port) {
        super(address, port);
    }

    Benchmark createBenchmark(int problemId) throws EvoXException {
        Config config;
        Args args = new Args();
        switch (problemId) {
            case 1:
                args.objs = "err&params";
                args.normalized_objectives = true;
                config = new Config("nb101", args);
                break;
            case 2:
                args.objs = "err&params&flops";
                args.normalized_objectives = true;
                config = new Config("nb101", args);
                break;
            case 3:
                args.fidelity = 90;
                args.objs = "err&params&flops";
                args.dataset = "cifar10";
                args.normalized_objectives = true;
                config = new Config("nats", args);
                break;
            case 4:
                args.fidelity = 90;
                args.objs = "err&params&flops&latency";
                args.dataset = "cifar10";
                args.normalized_objectives = true;
                config = new Config("nats", args);
                break;
            case 5:
                args.fidelity = 200;
                args.objs = "err&params&flops&edgegpu_latency";
                args.dataset = "cifar10";
                args.normalized_objectives = true;
                config = new Config("nb201", args);
                break;
            case 6:
                args.fidelity = 200;
                args.objs = "err&params&flops&edgegpu_latency&edgegpu_energy";
                args.dataset = "cifar10";
                args.normalized_objectives = true;
                config = new Config("nb201", args);
                break;
            case 7:
                args.fidelity = 200;
                args.objs = "err&params&flops&eyeriss_latency&eyeriss_energy&eyeriss_arithmetic_intensity";
                args.dataset = "cifar10";
                args.normalized_objectives = true;
                config = new Config("nb201", args);
                break;
            case 8:
                args.fidelity = 200;
                args.objs = "err&params&flops&edgegpu_latency&edgegpu_energy&eyeriss_latency&eyeriss_energy&eyeriss_arithmetic_intensity";
                args.dataset = "cifar10";
                args.normalized_objectives = true;
                config = new Config("nb201", args);
                break;
            case 9:
                args.objs = "err&params";
                args.normalized_objectives = false;
                config = new Config("darts", args);
                break;
            case 10:
                args.objs = "err&flops";
                args.normalized_objectives = false;
                config = new Config("darts", args);
                break;
            case 11:
                args.objs = "err&params&flops";
                args.normalized_objectives = false;
                config = new Config("darts", args);
                break;
            default:
                throw new EvoXException("No such problem id");
        }
        Benchmark bm = new Benchmark();
        bm.init(address, port, config);

        return bm;
    }
    
}