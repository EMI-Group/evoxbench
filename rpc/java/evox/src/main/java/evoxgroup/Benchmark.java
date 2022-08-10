package evoxgroup;

import java.net.Socket;
import java.net.UnknownHostException;
import java.io.*;
import com.google.gson.Gson;

import evoxgroup.Config;
import evoxgroup.EvoXException;

class CreateRequest {
    private String operation;
    private Config config;

    CreateRequest(Config config) {
        this.operation = "create";
        this.config = config;
    }
}

class CreateResponse {
    String status;
    String object_descriptor;
}

class SampleRequest {
    private String operation;
    private String object_descriptor;
    private int n_samples;

    SampleRequest(String object_descriptor, int n_samples) {
        this.operation = "sample";
        this.object_descriptor = object_descriptor;
        this.n_samples = n_samples;
    }
}

class SampleResponse {
    String status;
    int[][] result;
}

class EvaluateRequest {
    String operation;
    String object_descriptor;
    int[][] encoding;
    boolean true_eval;

    EvaluateRequest(String object_descriptor, int[][] encoding, boolean true_eval) {
        this.operation = "query";
        this.object_descriptor = object_descriptor;
        this.encoding = encoding;
        this.true_eval = true_eval;
    }
}

class EvaluateResponse {
    String status;
    float[][] result;
}

public class Benchmark {
    private String address;
    private int port;
    private String objId;
    private BufferedReader connIn;
    private OutputStreamWriter connOut;
    private boolean trueEval;
    private Gson gson;

    Benchmark() {
        this.trueEval = false;
    }

    protected void init(String address, int port, Config config) throws EvoXException {
        Socket socket;
        gson = new Gson();
        try {
            socket = new Socket(address, port);
            connIn = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            connOut = new OutputStreamWriter(socket.getOutputStream());
            connOut.write(gson.toJson(new CreateRequest(config)));
            connOut.write('\n');
            connOut.flush();
            CreateResponse response = gson.fromJson(connIn.readLine(), CreateResponse.class);
            if (!response.status.equals("ok")) {
                throw new EvoXException("Create failed");
            }
            objId = response.object_descriptor;
        } catch (UnknownHostException e) {
            throw new EvoXException("Unknown Host");
        } catch (IOException e) {
            throw new EvoXException("IO failed");
        }
    }

    public int[][] sample(int numSamples) throws EvoXException {
        try {
            connOut.write(gson.toJson(new SampleRequest(objId, numSamples)));
            connOut.write('\n');
            connOut.flush();

            SampleResponse response = gson.fromJson(connIn.readLine(), SampleResponse.class);
            if (!response.status.equals("ok")) {
                throw new EvoXException("Sample failed");
            }
            return response.result;
        } catch (IOException e) {
            throw new EvoXException("IO failed");
        }
    }

    public float[][] evaluate(int[][] encoding) throws EvoXException {
        try {
            connOut.write(gson.toJson(new EvaluateRequest(objId, encoding, trueEval)));
            connOut.write('\n');
            connOut.flush();

            EvaluateResponse response = gson.fromJson(connIn.readLine(), EvaluateResponse.class);
            if (!response.status.equals("ok")) {
                throw new EvoXException("Query failed");
            }
            return response.result;
        } catch (IOException e) {
            throw new EvoXException("IO failed");
        }
    }
}