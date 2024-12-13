from flask import Flask, render_template, request
import subprocess
import os

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home_index.html")  # Render the new homepage

@app.route("/simulation")
def simulation():
    return render_template("simulation.html", results=None)

@app.route("/run_simulation", methods=["POST"])
def run_simulation():
    simulation_type = request.form.get("simulation_type")

    # Map simulation types to scripts
    scripts = {
        "bilstm": "scripts/Predictive VM Allocation Simulator using Bidirectional LSTM.py",
        "linear": "scripts/Predictive VM Allocation Simulator using GRU + Linear Regression.py",
        "gru": "3. scripts/Predictive VM Allocation Simulator using GRU + Linear Regression.py"
    }

    script_to_run = scripts.get(simulation_type)

    if not script_to_run or not os.path.exists(script_to_run):
        return render_template("simulation.html", results=f"Invalid simulation type: {simulation_type}")

    try:
        # Execute the script and capture the output
        output = subprocess.check_output(["python", script_to_run], text=True)
        return render_template("simulation.html", results=output)
    except subprocess.CalledProcessError as e:
        return render_template("simulation.html", results=f"Error running simulation: {e}")


@app.route("/comparison")
def comparison():
    # Logic for Comparison
    return render_template("index.html")

@app.route("/testing")
def testing():
    return render_template("testing.html", results=None)

@app.route("/run_testing", methods=["POST"])
def run_testing():
    test_type = request.form.get("test_type")

    # Map test types to scripts
    scripts = {
        "prediction_allocation": "scripts/Prediction and Allocation Testing.py",
        "real_world_testing": "scripts/Real world testing.py",
        "unit_testing": "scripts/unit testing.py"
    }

    script_to_run = scripts.get(test_type)

    if not script_to_run or not os.path.exists(script_to_run):
        return render_template("testing.html", results=f"Invalid test type: {test_type}")

    try:
        # Execute the script and capture the output
        output = subprocess.check_output(["python", script_to_run], text=True)
        return render_template("testing.html", results=output)
    except subprocess.CalledProcessError as e:
        return render_template("testing.html", results=f"Error running test: {e}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=2000)

