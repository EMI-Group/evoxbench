package evoxgroup;
import java.util.Arrays;
import java.lang.Math;
import java.io.*;
import java.lang.String;
import com.google.gson.Gson;

class JavaEnvironmentReport {
    String jdk_version;
    String os;
}

class PythonEnvironmentReport {
    String python_version;
    String os;
}

class EnvironmentReport {
    JavaEnvironmentReport client;
    PythonEnvironmentReport server;

    EnvironmentReport() {
        client = new JavaEnvironmentReport();
        client.jdk_version = "17.0.3";
        client.os = "NixOS 22.11pre380684.83658b28fe6";

        server = new PythonEnvironmentReport();
        server.python_version = "3.10.4";
        server.os = "NixOS 22.11pre380684.83658b28fe6";
    }
}

class LatencyReport {
    float[] raw_data;
    float mean;
    float std;
    EnvironmentReport environment;

    private static float mean(float[] x) {
        float sum = 0;
        for(float elem : x) {
            sum += elem;
        }
        return sum / x.length;
    }

    private static float std(float[] x) {
        float mean = mean(x);
        float sum = 0;
        for(float elem : x) {
            sum += Math.pow(elem - mean, 2);
        }
        return (float)Math.sqrt(sum / (float)x.length);
    }

    LatencyReport(float[] raw_data) {
        this.raw_data = raw_data;
        this.mean = mean(raw_data);
        this.std = std(raw_data);
        environment = new EnvironmentReport();
    }
}

public class App {
    

    public static void main( String[] args ) {
        float[] latencies = new float[31];
        try {
            Benchmark bm;
            if(args[0].equals("nb201")) {
                C10MOP c10mop = new C10MOP("127.0.0.1", 9876);
                bm = c10mop.createBenchmark(5);
            }
            else if(args[0].equals("nb101")) {
                C10MOP c10mop = new C10MOP("127.0.0.1", 9876);
                bm = c10mop.createBenchmark(1);
            } else if(args[0].equals("nats")) {
                C10MOP c10mop = new C10MOP("127.0.0.1", 9876);
                bm = c10mop.createBenchmark(3);
            }
            else {
                IN1KMOP in1kmop = new IN1KMOP("127.0.0.1", 9876);
                bm = in1kmop.createBenchmark(7);
            }
            
            for(int i = 0; i < 31; i++){
                int[][] sample = bm.sample(100);
                long startTime = System.nanoTime();
                float[][] f = bm.evaluate(sample);
                long endTime = System.nanoTime();
                latencies[i] = (endTime - startTime) / (float)1e9;
            }
            LatencyReport report = new LatencyReport(latencies);
            PrintWriter out = new PrintWriter(String.format("query_latency_java_%s.json", args[0]));
            Gson gson = new Gson();
            System.out.println(gson.toJson(report));
            out.println(gson.toJson(report));
            out.close();
        }
        catch(EvoXException e) {
            e.printStackTrace();
        }
        catch(FileNotFoundException e) {
            e.printStackTrace();
        }
        System.out.println( "End!" );
    }
}
