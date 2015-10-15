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

    double[][] weights;

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
                makeSomeToast();

            }
        });
    }

    /**
     * Standard sigmoid function (logistic) used as activation function.
     *
     * @param a
     * @return
     */
    double sigmoid(double a) {
        return 1.0/(1+Math.exp(-a));
    }

    /**
     * Calculate the output of a single unit.
     *
     * @param prevLayer The input layer of that unit (previous layer).
     * @param weights   The input weights of the unit, including bias unit.
     * @return          The output of the unit (activated & weighted sum).
     */
    double calcOutput(double[] prevLayer, double[] weights) {
        double sum = 0;
        for (int idx = 0; idx < prevLayer.length; idx++) {
            sum += prevLayer[idx]*weights[idx];
        }
        sum += weights[weights.length-1];  // bias unit
        return sigmoid(sum);
    }

    void makeSomeToast() {
        Context context = getApplicationContext();
        CharSequence text = "Hello toast!";
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
