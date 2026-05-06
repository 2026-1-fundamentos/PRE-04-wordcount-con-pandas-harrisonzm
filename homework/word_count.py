"""Word count helpers built with pandas."""

import glob
import os
import string
import time

import pandas as pd


def copy_raw_files_to_input_folder(n):
    """Generate n copies of the raw files in the input folder."""
    input_directory = "files/input"

    if os.path.exists(input_directory):
        for file_path in glob.glob(f"{input_directory}/*"):
            os.remove(file_path)
    else:
        os.makedirs(input_directory)

    for file_path in glob.glob("files/raw/*"):
        with open(file_path, "r", encoding="utf-8") as file_handle:
            text = file_handle.read()

        raw_filename_with_extension = os.path.basename(file_path)
        raw_filename_without_extension = os.path.splitext(raw_filename_with_extension)[0]

        for index in range(1, n + 1):
            new_filename = f"{raw_filename_without_extension}_{index}.txt"
            with open(f"{input_directory}/{new_filename}", "w", encoding="utf-8") as output_file:
                output_file.write(text)


def load_input(input_directory):
    """Load every line from the input directory into a dataframe."""
    rows = []
    for file_path in glob.glob(f"{input_directory}/*"):
        with open(file_path, "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                rows.append({"file": file_path, "line": line})
    return pd.DataFrame(rows, columns=["file", "line"])


def preprocess_line(line):
    """Normalize a line and split it into words."""
    line = line.lower()
    line = line.translate(str.maketrans("", "", string.punctuation))
    line = line.replace("\n", "")
    return line.split()


def map_line(line):
    """Map a line to word-count pairs."""
    return [(word, 1) for word in preprocess_line(line)]


def mapper(sequence):
    """Mapper stage using pandas transformations."""
    if sequence.empty:
        return pd.DataFrame(columns=["word", "count"])

    dataframe = sequence.copy()
    dataframe["words"] = dataframe["line"].apply(preprocess_line)
    dataframe = dataframe.explode("words").dropna(subset=["words"])
    dataframe = dataframe.rename(columns={"words": "word"})
    dataframe["count"] = 1
    return dataframe[["word", "count"]]


def shuffle_and_sort(sequence):
    """Shuffle and sort stage."""
    return sequence.sort_values(by="word", kind="stable").reset_index(drop=True)


def compute_sum_by_group(group):
    """Sum the values for a grouped key."""
    key = group[0][0]
    total = sum(value for _, value in group)
    return key, total


def reducer(sequence):
    """Reducer stage."""
    if sequence.empty:
        return sequence

    return (
        sequence.groupby("word", as_index=False)["count"]
        .sum()
        .sort_values(by="word", kind="stable")
        .reset_index(drop=True)
    )


def create_directory(directory):
    """Create or clean the output directory."""
    if os.path.exists(directory):
        for file_path in glob.glob(f"{directory}/*"):
            os.remove(file_path)
    else:
        os.makedirs(directory)


def save_output(output_directory, sequence):
    """Save the reducer output."""
    output_path = f"{output_directory}/part-00000"
    sequence.to_csv(output_path, sep="\t", header=False, index=False)


def create_marker(output_directory):
    """Create the success marker file."""
    with open(f"{output_directory}/_SUCCESS", "w", encoding="utf-8") as output_file:
        output_file.write("")


def run_job(input_directory, output_directory):
    """Run the word count pipeline."""
    sequence = load_input(input_directory)
    sequence = mapper(sequence)
    sequence = shuffle_and_sort(sequence)
    sequence = reducer(sequence)
    create_directory(output_directory)
    save_output(output_directory, sequence)
    create_marker(output_directory)

if __name__ == "__main__":
    copy_raw_files_to_input_folder(n=1000)

    start_time = time.time()
    run_job("files/input", "files/output")
    end_time = time.time()

    print(f"Tiempo de ejecución: {end_time - start_time:.2f} segundos")

