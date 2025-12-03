import re
import ast
import re

#----------
#
#  Parse data written by mean_field_approx.py to experiment_data/mfa_xy_data.txt
#
#----------

def parse_sample_data(filename):
    # Parse the MFA data file into a list of samples.
    samples = []
    current_sample = {}
    current_dataset = None
    x_data = []
    y_data = []

    def parse_w1_all_runs_y_line(line):
        y_str = line[2:].strip()
        # Remove np.float64(...) wrappers
        y_str_clean = re.sub(r'np\.float64\(([^)]+)\)', r'\1', y_str)
        try:
            return ast.literal_eval(y_str_clean)
        except Exception as e:
            print(f"Error parsing w1 all runs y line: {e}")
            return []

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            # Start of new sample
            if line == "==New Sample==":
                if current_dataset and x_data:
                    current_sample[current_dataset] = {'x': x_data, 'y': y_data}
                    x_data, y_data = [], []
                if current_sample:
                    samples.append(current_sample)
                    current_sample = {}
                current_dataset = None
                continue

            # Detect Number of nodes line
            if line.startswith("Number of nodes:"):
                try:
                    current_sample['Number of nodes'] = int(line.split(":")[1].strip())
                except Exception as e:
                    print(f"Error parsing Number of nodes: {e}")
                    current_sample['Number of nodes'] = None
                continue

            # Detect true adherence proportion
            if line.startswith("Adhering proportion:"):
                try:
                    current_sample['Adhering proportion'] = float(line.split(":")[1].strip())
                except Exception as e:
                    print(f"Error parsing Adhering proportion: {e}")
                    current_sample['Adhering proportion'] = None
                continue

            # Detect dataset name lines
            if line.startswith(("SIR Infections", "Dynamic degree", "w1 True", "w1 Estimated",
                                "w1 all runs", "w2 True", "w2 Estimated",
                                "Informed and Infected", "Given Newly Infected Ratio", "Informed")):
                if current_dataset and x_data:
                    current_sample[current_dataset] = {'x': x_data, 'y': y_data}
                    x_data, y_data = [], []
                current_dataset = line.split(':')[0].strip()
                continue

            # Parse x-data
            if line.startswith("x:"):
                try:
                    x_data = [float(x) for x in line[2:].split(',') if x]
                except ValueError as e:
                    print(f"Error parsing x-data: {e}")
                    x_data = []

            # Parse y-data
            elif line.startswith("y:"):
                if current_dataset == 'w1 all runs':
                    y_data = parse_w1_all_runs_y_line(line)
                elif current_dataset == 'Informed':
                    # Evaluate as Python object (list of lists)
                    try:
                        y_data = ast.literal_eval(line[2:].strip())
                    except Exception as e:
                        print(f"Error parsing Informed: {e}")
                        y_data = []
                else:
                    raw = line[2:].strip()
                    try:
                        # If starts with '[', evaluate as a Python object (nested lists)
                        if raw.startswith('['):
                            y_data = ast.literal_eval(raw)
                        else:
                            y_data = [float(y) for y in raw.split(',') if y]
                    except Exception as e:
                        print(f"Error parsing y-data: {e}")
                        y_data = []

        # Save last dataset of the last sample
        if current_dataset and x_data:
            current_sample[current_dataset] = {'x': x_data, 'y': y_data}
        if current_sample:
            samples.append(current_sample)

    return samples

# Plot description: mfa_xy should have many of 1 config, many of another (split = False in mean_field_approx.py), 
#   and this will plot the SIR curves and w1 True vs Estimated for each half (set of samples).
def plot_halves(samples):
    from itertools import cycle
    import matplotlib.pyplot as plt

    n_samples = len(samples)
    mid_point = (n_samples + 1) // 2
    first_half = samples[:mid_point]
    second_half = samples[mid_point:]

    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

    # Plot first half
    if first_half:
        # SIR infections
        plt.figure(figsize=(10, 6))
        for i, sample in enumerate(first_half, 1):
            color = next(colors)
            plt.plot(sample['SIR Infections']['x'], sample['SIR Infections']['y'],
                     label=f'Sample {i}', color=color)
        plt.xlabel('Time')
        plt.ylabel('SIR Infections')
        plt.title('First Half: SIR Infections')
        plt.grid(True)
        plt.legend()
        plt.show()

        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

        # w1 True and Estimated
        plt.figure(figsize=(10, 6))
        for i, sample in enumerate(first_half, 1):
            color = next(colors)
            if 'w1 True (Mean Node Degree)' in sample:
                plt.plot(sample['w1 True (Mean Node Degree)']['x'], sample['w1 True (Mean Node Degree)']['y'],
                         label=f'Sample {i} w1 True', color=color, linestyle='-')
            plt.plot(sample['w1 Estimated']['x'], sample['w1 Estimated']['y'],
                     label=f'Sample {i} w1 Estimated', color=color, linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Mean Node Degree')
        plt.title('First Half: w1 True vs Estimated')
        plt.grid(True)
        plt.legend()
        plt.show()

    # Plot second half
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

    if second_half:
        # SIR infections
        plt.figure(figsize=(10, 6))
        for i, sample in enumerate(second_half, mid_point + 1):
            color = next(colors)
            plt.plot(sample['SIR Infections']['x'], sample['SIR Infections']['y'],
                     label=f'Sample {i}', color=color)
        plt.xlabel('Time')
        plt.ylabel('SIR Infections')
        plt.title('Second Half: SIR Infections')
        plt.grid(True)
        plt.legend()
        plt.show()

        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])

        # w1 True and Estimated
        plt.figure(figsize=(10, 6))
        for i, sample in enumerate(second_half, mid_point + 1):
            color = next(colors)
            if 'w1 True (Mean Node Degree)' in sample:
                plt.plot(sample['w1 True (Mean Node Degree)']['x'], sample['w1 True (Mean Node Degree)']['y'],
                         label=f'Sample {i} w1 True', color=color, linestyle='-')
            plt.plot(sample['w1 Estimated']['x'], sample['w1 Estimated']['y'],
                     label=f'Sample {i} w1 Estimated', color=color, linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Mean Node Degree')
        plt.title('Second Half: w1 True vs Estimated')
        plt.grid(True)
        plt.legend()
        plt.show()