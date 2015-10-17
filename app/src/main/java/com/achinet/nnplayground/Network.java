package com.achinet.nnplayground;

import android.os.AsyncTask;
import android.widget.TextView;

/**
 * Created by achilleas on 16/10/15.
 */
public class Network extends AsyncTask<Double[], Integer, Double> {

    TextView progressTextView;

    // Number of nodes per layer
    int numInputs;
    int numNodesHL1;
    int numNodesHL2;
    int numNodesOL;

    // Weight arrays
    double[][] weightsHL1;
    double[][] weightsHL2;
    double[][] weightsOL;

    // Intermediate output arrays
    double[] outputsHL1;
    double[] outputsHL2;
    double[] outputsNet;

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

        outputsHL1 = new double[numNodesHL1];
        outputsHL2 = new double[numNodesHL2];
        outputsNet = new double[numNodesOL];
    }

    void trainNN(double[] inputs, double[] targetOutputs) {
        double[] outputs = forwardPass(inputs);
        double[] errors = arrayDiff(outputs, targetOutputs);
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
        outputsHL1 = calcLayerOutput(inputs, weightsHL1);
        outputsHL2 = calcLayerOutput(outputsHL1, weightsHL2);
        outputsNet = calcLayerOutput(outputsHL2, weightsOL);
        return outputsNet;
    }

    /**
     * Run a backward pass of the network, updating the weights accordingly.
     *
     * @param errors The errors of the last forward pass.
     */
    void backwardPass(double[] errors) {
        // output layer deltas
        double[] deltasOL = new double[numNodesOL];
        for (int idx = 0; idx < numNodesOL; idx++) {
            deltasOL[idx] = outputsNet[idx]*(1-outputsNet[idx])*errors[idx];
        }
        // hidden layer 2 deltas
        double[] deltasHL2 = new double[numNodesHL2];
        for (int idx = 0; idx < numNodesHL2; idx++) {
            double weightDeltaSum = 0;
            for (int jdx = 0; jdx < numNodesOL; jdx++) {
                weightDeltaSum += weightsOL[jdx][idx]*deltasOL[jdx];
            }
            deltasHL2[idx] = outputsHL2[idx]*(1-outputsHL2[idx])*weightDeltaSum;
        }
        // hidden layer 1 deltas
        double[] deltasHL1 = new double[numNodesHL1];
        for (int idx = 0; idx < numNodesHL1; idx++) {
            double weightDeltaSum = 0;
            for (int jdx = 0; jdx < numNodesHL2; jdx++) {
                weightDeltaSum += weightsHL2[jdx][idx]*deltasHL2[jdx];
            }
            deltasHL1[idx] = outputsHL1[idx]*(1-outputsHL1[idx])*weightDeltaSum;
        }
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
     * 1/(1+e^(-x))
     *
     * @param x
     * @return
     */
    double sigmoid(double x) {
        return 1.0/(1+Math.exp(-x));
    }

    void setScreenText(String text) {
        progressTextView.setText(text);
    }

    void appendScreenText(String text) {
        String existingText = (String)progressTextView.getText();
        setScreenText(existingText + text);
    }

    /**
     * Simple method to return a string representation of an array's elements.
     *
     * @param array
     * @return
     */
    String arrayToString(double[] array) {
        StringBuilder str = new StringBuilder("[");
        for (double v : array) {
            str.append(v+", ");
        }
        str.delete(str.length()-2, str.length());
        str.append("]");
        return str.toString();
    }

    /**
     * Return the element-wise difference between two arrays. No checks are
     * performed to make sure the arrays are of the same size.
     *
     * @param one
     * @param two
     * @return
     */
    double[] arrayDiff(double[] one, double[] two) {
        double[] result = new double[one.length];
        for (int idx = 0; idx < one.length; idx++) {
            result[idx] = one[idx]-two[idx];
        }
        return result;
    }

    double[] objectToPrimitive(Double[] objArray) {
        double[] primArray = new double[objArray.length];
        for (int idx = 0; idx < objArray.length; idx++) {
            primArray[idx] = objArray[idx];
        }
        return primArray;
    }

    // AsyncTask methods
    @Override
    protected Double doInBackground(Double[]... args) {
        double[] inputs = objectToPrimitive(args[0]);
        double[] targetOutputs = objectToPrimitive(args[1]);
        int nIter = 20000;
        long startTime = System.currentTimeMillis();
        for (int n = 0; n < nIter; n++) {
            trainNN(inputs, targetOutputs);
            publishProgress(n + 1, nIter);
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
