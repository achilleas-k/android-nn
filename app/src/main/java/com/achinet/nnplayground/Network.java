package com.achinet.nnplayground;

import android.os.AsyncTask;
import android.widget.ProgressBar;
import android.widget.TextView;

/**
 * Created by achilleas on 16/10/15.
 */
public class Network extends AsyncTask<Integer, Double, Double> {

    TextView progressTextView;
    ProgressBar progressbar;

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

    double[][] inputs;
    double[][] targetOutputs;

    // TODO: parameterize
    double learningRate = 0.2;

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

    public void setProgressBar(ProgressBar pb) {
        progressbar = pb;
        progressbar.setMax(100);
    }
    public void setData(double[][] in, double[][] out) {
        inputs = in;
        targetOutputs = out;
    }

    void initNN() {
        weightsHL1 = rand2D(numNodesHL1, numInputs+1);   // input to HL1 weights
        weightsHL2 = rand2D(numNodesHL2, numNodesHL1+1); // HL1 to HL2 weights
        weightsOL  = rand2D(numNodesOL, numNodesHL2+1);  // HL2 to output weights

        outputsHL1 = new double[numNodesHL1];
        outputsHL2 = new double[numNodesHL2];
        outputsNet = new double[numNodesOL];
    }

    /**
     * Run a single training iteration (epoch) across all the data.
     *
     * @return Mean square error of the iteration.
     */
    double trainNN() {
        double mse = 0;
        for (int didx = 0; didx < inputs.length; didx++) {
            double[] outputs = forwardPass(inputs[didx]);
            double[] errors = arrayDiff(outputs, targetOutputs[didx]);
            backwardPass(errors, inputs[didx]);
            for (int idx = 0; idx < numNodesOL; idx++) {
                mse += Math.pow(errors[idx], 2);
            }
            mse /= numNodesOL;
        }
        return mse/inputs.length;
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
    void backwardPass(double[] errors, double[] inputs) {
        // TODO: Can be simplified (which would also generalise)
        // TODO: Include momentum
        // output layer deltas
        double[] deltasOL = new double[numNodesOL];
        for (int idx = 0; idx < numNodesOL; idx++) {
            deltasOL[idx] = outputsNet[idx]*(1-outputsNet[idx])*errors[idx];
        }
        // hidden layer 2 deltas
        double[] deltasHL2 = new double[numNodesHL2];
        for (int nh2 = 0; nh2 < numNodesHL2; nh2++) {
            double weightDeltaSum = 0;
            for (int nol = 0; nol < numNodesOL; nol++) {
                weightDeltaSum += weightsOL[nol][nh2]*deltasOL[nol];
            }
            deltasHL2[nh2] = outputsHL2[nh2]*(1-outputsHL2[nh2])*weightDeltaSum;
        }
        // hidden layer 1 deltas
        double[] deltasHL1 = new double[numNodesHL1];
        for (int nh1 = 0; nh1 < numNodesHL1; nh1++) {
            double weightDeltaSum = 0;
            for (int nh2 = 0; nh2 < numNodesHL2; nh2++) {
                weightDeltaSum += weightsHL2[nh2][nh1]*deltasHL2[nh2];
            }
            deltasHL1[nh1] = outputsHL1[nh1]*(1-outputsHL1[nh1])*weightDeltaSum;
        }

        // hidden layer 1 weight updates
        for (int nh1 = 0; nh1 < numNodesHL1; nh1++) {
            for (int nin = 0; nin < numInputs; nin++) {
                weightsHL1[nh1][nin] -= learningRate*deltasHL1[nh1]*inputs[nin];
            }
            // bias unit
            weightsHL1[nh1][numInputs] -= learningRate*deltasHL1[nh1];
        }
        // hidden layer 2 weight updates
        for (int nh2 = 0; nh2 < numNodesHL2; nh2++) {
            for (int nh1 = 0; nh1 < numNodesHL1; nh1++) {
                weightsHL2[nh2][nh1] -= learningRate*deltasHL2[nh2]*outputsHL1[nh1];
            }
            // bias unit
            weightsHL2[nh2][numNodesHL1] -= learningRate*deltasHL2[nh2];
        }
        // output layer weight updates
        for (int nol = 0; nol < numNodesOL; nol++) {
            for (int nh2 = 0; nh2 < numNodesHL2; nh2++) {
                weightsOL[nol][nh2] -= learningRate*deltasOL[nol]*outputsHL2[nh2];
            }
            // bias unit
            weightsOL[nol][numNodesHL2] -= learningRate*deltasOL[nol];
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
    // AsyncTask methods
    @Override
    protected Double doInBackground(Integer... args) {
        double nIter = args[0];
        long startTime = System.currentTimeMillis();
        double mse;
        for (int n = 0; n < nIter; n++) {
            mse = trainNN();
            publishProgress(n + 1.0, nIter, mse);
        }
        return 1.0*(System.currentTimeMillis()-startTime)/nIter;
    }

    @Override
    protected void onProgressUpdate(Double... progress) {
        double progressPerc = 100*progress[0]/progress[1];
        String progressReport = "Progress: "+progress[0]+
                "/"+progress[1]+" iterations"+" ("+progressPerc+" %)";
        progressReport += "\nMSE: "+progress[2];
        setScreenText(progressReport);
        progressbar.setProgress((int)progressPerc);
    }

    @Override
    protected void onPostExecute(Double duration) {
        appendScreenText("\nAverage iteration runtime: "+duration+" ms");
        appendScreenText("\nNumber of records used in training: "+inputs.length);
        String finalOutputs = "\nFinal outputs:\n";
        for (int idx = 0; idx < inputs.length; idx++) {
            forwardPass(inputs[idx]);
            finalOutputs += arrayToString(inputs[idx])+" : "+arrayToString(outputsNet)+"\n";
        }
        appendScreenText(finalOutputs);
    }
}
