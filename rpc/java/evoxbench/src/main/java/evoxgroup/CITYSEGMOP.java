package evoxgroup;
import evoxgroup.EvoXException;
import evoxgroup.Benchmark;
import evoxgroup.BenchmarkFactory;
import evoxgroup.Config;
import evoxgroup.Args;

public class CITYSEGMOP extends BenchmarkFactory {
    CITYSEGMOP(String address, int port) {
        super(address, port);
    }

    Benchmark createBenchmark(int problemId) throws EvoXException {
        Config config;
        Args args = new Args();

        switch (problemId) {
            case 1:
                args.objs = "err&h1_latency";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 2:
                args.objs = "err&h1_latency&flops";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 3:
                args.objs = "err&h1_latency&params";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 4:
                args.objs = "err&h1_latency&h1_energy_consumption&flops";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 5:
                args.objs = "err&h1_latency&h1_energy_consumption&flops&params";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 6:
                args.objs = "err&h2_latency";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 7:
                args.objs = "err&h2_latency&flops";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 8:
                args.objs = "err&h2_latency&params";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 9:
                args.objs = "err&h2_latency&h2_energy_consumption&flops";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 10:
                args.objs = "err&h1_latency&h1_energy_consumption&flops&params";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 11:
                args.objs = "err&h1_latency&h2_latency";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 12:
                args.objs = "err&h1_latency&h2_latency&h1_energy_consumption&h2_energy_consumption";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 13:
                args.objs = "err&h1_latency&h2_latency&h1_energy_consumption&h2_energy_consumption&flops";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 14:
                args.objs = "err&h1_latency&h2_latency&h1_energy_consumption&h2_energy_consumption&params";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 15:
                args.objs = "err&h1_latency&h2_latency&h1_energy_consumption&h2_energy_consumption&flops&params";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            default:
                throw new EvoXException("No such problem id");
        }
        Benchmark bm = new Benchmark();
        bm.init(address, port, config);

        return bm;
    }
}