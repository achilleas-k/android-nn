package com.achinet.nnplayground;

import android.content.Context;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.design.widget.Snackbar;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.Toast;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    double[][] weightsHL1;
    double[][] weightsHL2;
    double[][] weightsOL;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                        .setAction("Action", null).show();
                initNN();
                initToast();

            }
        });
    }

    void initNN() {
        int numInputs = 10;
        int numNodesHL1 = 20;
        int numNodesHL2 = 20;
        int numNodesOL = 1;
        weightsHL1 = rand2D(numNodesHL1, numInputs+1);   // input to HL1 weights
        weightsHL2 = rand2D(numNodesHL2, numNodesHL1+1); // HL1 to HL2 weights
        weightsOL  = rand2D(numNodesOL, numNodesHL2+1);  // HL2 to output weights
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
        double[] valuesHL1 = calcLayerOutput(inputs, weightsHL1);
    }

    /**
     * Calculate the output values of the nodes on a whole layer.
     *
     * @param valuesPrevLayer   Values of the previous layer, or input layer for HL1.
     * @param weights           Weights between the current and previous layer.
     * @return                  An array of the output value of each node.
     */
    static double[] calcLayerOutput(double[] valuesPrevLayer, double[][] weights) {
        double[] values = new double[weights.length];
        for (int idx = 0; idx < weights.length; idx++) {
            values[idx] = calcNodeOutput(valuesPrevLayer, weights[idx];
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
    static double calcNodeOutput(double[] prevLayer, double[] weights) {
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
    static double sigmoid(double a) {
        return 1.0/(1+Math.exp(-a));
    }

    void initToast() {
        Context context = getApplicationContext();
        CharSequence text = "NN initialised";
        int duration = Toast.LENGTH_SHORT;

        Toast toast = Toast.makeText(context, text, duration);
        toast.show();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }
}
