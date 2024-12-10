# Simulated real-world dataset (CPU usage over time)
real_world_data = [45, 50, 80, 30, 70, 55, 90, 35, 60]

# Traditional Approach
def traditional_simulation(data):
    time = 0
    energy = 0
    for usage in data:
        time += usage * 0.8
        energy += usage * 1.2
    return time, energy

# ML-Based Approach
def ml_simulation(data, predicted_usage):
    time = 0
    energy = 0
    for actual, predicted in zip(data, predicted_usage):
        time += abs(predicted - actual) * 0.7 + 5
        energy += predicted * 1.1
    return time, energy


predicted_data = [50, 55, 75, 35, 65, 60, 85, 40, 55]

trad_time, trad_energy = traditional_simulation(real_world_data)
ml_time, ml_energy = ml_simulation(real_world_data, predicted_data)

optimization = ((trad_time + trad_energy) - (ml_time + ml_energy)) / (trad_time + trad_energy) * 100

print(f"Traditional - Time: {trad_time:.2f}, Energy: {trad_energy:.2f}")
print(f"ML-Based - Time: {ml_time:.2f}, Energy: {ml_energy:.2f}, Optimization: {optimization:.2f}%")