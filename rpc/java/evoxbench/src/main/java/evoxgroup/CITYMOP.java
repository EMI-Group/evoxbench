package evoxgroup;


public class CITYMOP extends BenchmarkFactory {
    CITYMOP(String address, int port) {
        super(address, port);
    }

    Benchmark createBenchmark(int problemId) throws EvoXException {
        Config config;
        Args args = new Args();

        switch (problemId) {
            case 1:
                args.objs = "params&flops&mIoU&latency&FPS";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 2:
                args.objs = "params&mIoU&latency&FPS";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 3:
                args.objs = "flops&mIoU&latency&FPS";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 4:
                args.objs = "params&flops&mIoU";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 5:
                args.objs = "mIoU&latency&FPS";
                args.normalized_objectives = false;
                config = new Config("mosegnas", args);
                break;
            case 6:
                args.objs = "mIoU";
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