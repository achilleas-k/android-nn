package com.achinet.nnplayground;

import android.content.Context;
import android.os.AsyncTask;
import android.util.Log;
import android.widget.TextView;

/**
 * Created by achilleas on 16/10/15.
 */
public class Network extends AsyncTask<Double, Integer, Double> {

    TextView progressTextView;

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
        initNN();
    }

    public void setProgressTextView(TextView tv) {
        progressTextView = tv;
    }

    void initNN() {
        weightsHL1 = rand2D(numNodesHL1, numInputs+1);   // input to HL1 weights
        weightsHL2 = rand2D(numNodesHL2, numNodesHL1+1); // HL1 to HL2 weights
        weightsOL  = rand2D(numNodesOL, numNodesHL2+1);  // HL2 to output weights
    }

    void trainNN(double[] inputs) {
        double[] outputs = forwardPass(inputs);
        double[] errors = new double[0];
        backwardPass(errors);
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
     * Run a backward pass of the network, updating the weights accordingly.
     *
     * @param errors The MSE of the last forward pass.
     * @return
     */
    double[] backwardPass(double[] errors) {
        return new double[0];
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

    void setScreenText(String text) {
        progressTextView.setText(text);
    }

    void appendScreenText(String text) {
        String existingText = (String)progressTextView.getText();
        setScreenText(existingText+text);
    }

    String arrayToString(double[] array) {
        StringBuilder str = new StringBuilder("[");
        for (double v : array) {
            str.append(v+", ");
        }
        str.delete(str.length()-2, str.length());
        str.append("]");
        return str.toString();
    }


    // AsyncTask methods
    @Override
    protected Double doInBackground(Double... inputValues) {
        double[] inputs = new double[inputValues.length];
        for (int idx = 0; idx < inputValues.length; idx++) {
            inputs[idx] = inputValues[idx];
        }
        int nIter = 20000;
        long startTime = System.currentTimeMillis();
        for (int n = 0; n < nIter; n++) {
            trainNN(inputs);
            publishProgress(n+1, nIter);
        }
        return 1.0*(System.currentTimeMillis()-startTime)/nIter;
    }

    @Override
    protected void onProgressUpdate(Integer... progress) {
        float progressPerc = 100f*progress[0]/progress[1];
        String progressReport = "Progress: "+progress[0]+
                "/"+progress[1]+" iterations"+" ("+progressPerc+" %)";
        setScreenText(progressReport);
    }

    @Override
    protected void onPostExecute(Double duration) {
        appendScreenText("\nAverage iteration runtime: "+duration+" ms");
    }
}
