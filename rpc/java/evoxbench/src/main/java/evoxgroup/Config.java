package evoxgroup;

import evoxgroup.Args;

public class Config {
    private String name;
    private Args args;

    Config(String name, Args args) {
        this.name = name;
        this.args = args;
    }
}