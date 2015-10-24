package com.achinet.nnplayground;

import android.content.Context;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.support.design.widget.FloatingActionButton;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {

    Network network;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        loadConfig();

        FloatingActionButton fab = (FloatingActionButton) findViewById(R.id.fab);
        fab.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //Snackbar.make(view, "Replace with your own action", Snackbar.LENGTH_LONG)
                //        .setAction("Action", null).show();
                network = new Network(2, 200, 200, 1);
                network.setProgressTextView((TextView)findViewById(R.id.screen_text));
                double[][] inputs = new double[][]{
                        {0, 0},
                        {0, 1},
                        {1, 0},
                        {1, 1}};
                double[][] outputs = new double[][]{
                        {0}, {1}, {1}, {0}};
                network.setData(inputs, outputs);
                makeToast("NN initialised");
                network.execute(2000);
            }
        });
    }

    void makeToast(String text) {
        Context context = getApplicationContext();
        int duration = Toast.LENGTH_SHORT;

        Toast toast = Toast.makeText(context, text, duration);
        toast.show();
    }

    void loadConfig() {
        SharedPreferences sharedPref = getSharedPreferences(
                getString(R.string.preference_file_key), Context.MODE_PRIVATE);
        resetConfig(sharedPref);
        if (!sharedPref.contains("initialised")) initConfig(sharedPref);
        ((TextView)findViewById(R.id.nHL1)).setText(""+sharedPref.getInt("layer_one_size", 200));
        ((TextView)findViewById(R.id.nHL2)).setText(""+sharedPref.getInt("layer_two_size", 200));
        ((TextView)findViewById(R.id.maxIter)).setText("" + sharedPref.getInt("max_iterations", 100));
    }

    void saveConfig() {
        SharedPreferences.Editor editor = getSharedPreferences(
                getString(R.string.preference_file_key), Context.MODE_PRIVATE).edit();
        editor.putInt("layer_one_size",
                Integer.parseInt(((TextView) findViewById(R.id.nHL1)).getText().toString()));
        editor.putInt("layer_two_size",
                Integer.parseInt(((TextView) findViewById(R.id.nHL2)).getText().toString()));
        editor.putInt("max_iterations",
                Integer.parseInt(((TextView) findViewById(R.id.maxIter)).getText().toString()));
    }

    int getMainScreenInt(int id) {
        return Integer.parseInt(((TextView) findViewById(id)).getText().toString());
    }

    /**
     * Temporary method for clearing prefs file each run until the settings are finalised.
     * @param sharedPref
     */
    void resetConfig(SharedPreferences sharedPref) {
        sharedPref.edit().clear();
    }

    void initConfig(SharedPreferences sharedPref) {
        SharedPreferences.Editor editor = sharedPref.edit();
        editor.putString("initialised", "-");
        editor.putInt("layer_one_size", 200);
        editor.putInt("layer_two_size", 200);
        editor.putInt("max_iterations", 100);
        editor.commit();
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
