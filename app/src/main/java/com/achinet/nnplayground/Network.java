package com.achinet.nnplayground;

import android.util.Log;

import java.util.concurrent.Executor;

/**
 * Created by achilleas on 16/10/15.
 */
public class Network implements Executor {

    Runnable trainTask;

    double[][] weightsHL1;
    double[][] weightsHL2;
    double[][] weightsOL;
    int numInputs;
    int numNodesHL1;
    int numNodesHL2;
    int numNodesOL;

    public Network(int nIn, int nH1, int nH2, int nOut) {
        numInputs = nIn;
        numNodesHL1 = nH1;
        numNodesHL2 = nH2;
        numNodesOL = nOut;
        trainTask = new ForwardPass();
        initNN();
    }

    void initNN() {
        weightsHL1 = rand2D(numNodesHL1, numInputs+1);   // input to HL1 weights
        weightsHL2 = rand2D(numNodesHL2, numNodesHL1+1); // HL1 to HL2 weights
        weightsOL  = rand2D(numNodesOL, numNodesHL2+1);  // HL2 to output weights
    }

    void trainNN() {
        double[] inputs = new double[]{0.2, 0.213, 0.111, 0.98, 0.8, 1.0, 0, 0.3, 0.567, 0.9};
        double[] outputs = forwardPass(inputs);
        int nIter = 100;
        long startTime = System.currentTimeMillis();
        for (int n = 0; n < nIter; n++) {
            forwardPass(inputs);
        }
        double duration = 1.0*(System.currentTimeMillis()-startTime)/nIter;
    }

    double[][] rand2D(int m, int n) {
        double[][] array = new double[m][n];
        for (int idx = 0; idx < m; idx++) {
            for (int jdx = 0; jdx < n; jdx++) {
                array[idx][jdx] = Math.random()*2-1; // range [-1, 1)
            }
        }
        return array;
    }

    /**
     * Run a forward pass of the network, using the provided inputs. The weights
     * must already be set.
     *
     * @param inputs    Network input vector.
     * @return          The value(s) of the output layer.
     */
    double[] forwardPass(double[] inputs) {
        return calcLayerOutput(calcLayerOutput(calcLayerOutput(inputs, weightsHL1), weightsHL2), weightsOL);
    }

    /**
     * Calculate the output values of the nodes on a whole layer.
     *
     * @param valuesPrevLayer   Values of the previous layer, or input layer for HL1.
     * @param weights           Weights between the current and previous layer.
     * @return                  An array of the output value of each node.
     */
    double[] calcLayerOutput(double[] valuesPrevLayer, double[][] weights) {
        double[] values = new double[weights.length];
        for (int idx = 0; idx < weights.length; idx++) {
            values[idx] = calcNodeOutput(valuesPrevLayer, weights[idx]);
        }
        return values;
    }

    /**
     * Calculate the output of a single unit.
     *
     * @param prevLayer The input layer of that unit (previous layer).
     * @param weights   The input weights of the unit, including bias unit.
     * @return          The output of the unit (activated & weighted sum).
     */
    double calcNodeOutput(double[] prevLayer, double[] weights) {
        double sum = 0;
        for (int idx = 0; idx < prevLayer.length; idx++) {
            sum += prevLayer[idx]*weights[idx];
        }
        sum += weights[weights.length-1];  // bias unit
        return sigmoid(sum);
    }

    /**
     * Standard sigmoid function (logistic) used as activation function.
     * 1/(1+e^(-a))
     *
     * @param a
     * @return
     */
    double sigmoid(double a) {
        return 1.0/(1+Math.exp(-a));
    }


    public void execute(Runnable r) {
        r.run();
    }

    public class ForwardPass implements Runnable {
        public void run() {
            trainNN();
            Log.d("Network", "ForwardPass complete");
        }
    }

    public class BackwardPass implements Runnable {
        public void run() {
        }
    }
}