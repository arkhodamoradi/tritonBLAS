#!/usr/bin/env python3
import yaml
import argparse
import torch
import csv


def str_to_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert a string representation of a dtype to the corresponding torch.dtype.

    Args:
        dtype_str (str): The string representation of the dtype (e.g., "torch.float32").

    Returns:
        torch.dtype: The corresponding torch dtype.
    """
    # Remove the 'torch.' prefix if it exists
    dtype_str = dtype_str.replace("torch.", "")
    try:
        return getattr(torch, dtype_str)
    except AttributeError:
        raise ValueError(
            f"Invalid dtype string: '{dtype_str}'. Available options are: {', '.join([attr for attr in dir(torch) if isinstance(getattr(torch, attr), torch.dtype)])}"
        )


def bench_matmul(input_yaml: str):
    # Load benchmark cases from the YAML file
    with open(input_yaml, "r") as f:
        dataset = yaml.safe_load(f)

    benchmark_results = []
    # Convert the dataset cases into tuples: (m, n, k, in_dtype, out_dtype)
    dataset_tuples = [
        (
            case["m"],
            case["n"],
            case["k"],
            str_to_dtype(case["in_dtype"]),
            str_to_dtype(case["out_dtype"]),
            case["transA"],
            case["transB"],
        )
        for case in dataset
    ]

    # Iterate over all benchmark cases
    for m, n, k, in_dtype, out_dtype, transA, transB in dataset_tuples:
        # Initialize the matrices A and B with appropriate dimensions based on transA and transB
        if transA == "T":
            A_size = (m, k)  # A is MxK
        else:
            A_size = (k, m)  # A is KxM (we will later transpose it with .T)

        if transB == "T":
            B_size = (k, n)  # B is KxN
        else:
            B_size = (n, k)  # B is NxK (we will later transpose it with .T)

        # Initialize tensors with the appropriate dimensions
        A = torch.randn(*A_size, device="cuda", dtype=in_dtype)
        B = torch.randn(*B_size, device="cuda", dtype=in_dtype)

        # Apply transpose on A or B if necessary (only needed for "N" case)
        if transA == "N":
            A = A.T  # Apply transpose to A if transA is "N"

        if transB == "N":
            B = B.T  # Apply transpose to B if transB is "N"

        # Initialize tensors with the appropriate dimensions

        # Warm-up iterations
        for _ in range(20):
            _ = torch.matmul(A, B)

        # Benchmark the torch.matmul over 10 repetitions using CUDA events for timing.
        iterations = 10
        times = []
        for _ in range(iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            _ = torch.matmul(A, B)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)  # time in milliseconds
            times.append(elapsed_ms)

        # Calculate mean execution time (ms) and derive performance in TFLOPS.
        mean_ms = sum(times) / len(times)
        # Compute FLOPS count: 2 * m * n * k operations (each multiply-add counts as 2 operations) scaled to tera (1e-12)
        flops = 2 * m * n * k * 1e-12
        tflops = flops / (mean_ms * 1e-3)

        print(
            f"m={m}, n={n}, k={k}, in_dtype={in_dtype}, out_dtype={out_dtype} perf={tflops}"
        )

        # Calculate bytes processed: considering both A, B, and the output tensor.
        bytes_fn = lambda: (A.element_size() * (m * k + n * k)) + (
            m * n * A.element_size()
        )

        # Collect the metrics in a dictionary for later CSV output.
        metrics = {
            "m": m,
            "n": n,
            "k": k,
            "mnk": m * n * k,
            "bytes": bytes_fn(),
            "flops": flops,
            "tflops": tflops,
            "in_dtype": str(in_dtype),
            "out_dtype": str(out_dtype),
            "transA": transA,
            "transB": transB,
        }
        benchmark_results.append(metrics)

    return benchmark_results


def write_csv(filename: str, results):
    """Write the benchmark results to a CSV file."""
    fieldnames = [
        "m",
        "n",
        "k",
        "mnk",
        "bytes",
        "flops",
        "tflops",
        "in_dtype",
        "out_dtype",
        "transA",
        "transB",
    ]
    with open(filename, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"Benchmark results saved to '{filename}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark torch.matmul performance and optionally output performance metrics to a CSV file."
    )
    parser.add_argument(
        "--input-yaml",
        type=str,
        default="matmul_random.yaml",
        help="Input YAML file containing benchmark cases (default: ./matmul_random.yaml).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Filename for CSV output (if not specified, CSV output is disabled).",
    )
    args = parser.parse_args()

    benchmark_results = bench_matmul(args.input_yaml)

    if args.output_csv:
        write_csv(args.output_csv, benchmark_results)
