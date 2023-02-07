package evoxgroup;

class Args {
    String objs;
    boolean normalized_objectives;
    Integer fidelity;
    String dataset;

    Args() {}

    Args(String objs, boolean normalized_objectives, int fidelity, String dataset) {
        this.objs = objs;
        this.normalized_objectives = normalized_objectives;
        this.fidelity = fidelity;
        this.dataset = dataset;
    }
}