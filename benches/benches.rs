use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion};
use adam::adam_rs::{adam_basic, adam_optimized};

fn run_adam_benchmarks(c: &mut Criterion) {
    // --- Setup ---
    // Define the parameters for the benchmark run
    let alpha = 0.001;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;
    let n_steps = 100; // A fixed number of steps for a consistent benchmark
    let convergence_tol = 1e-6;
    let n_params = 128; // The dimensionality of our problem

    // The gradient of f(θ) = Σθᵢ², which is ∇f(θ) = 2θ
    let grad_fn = |theta: &[f64]| -> Vec<f64> {
        theta.iter().map(|&th| 2.0 * th).collect()
    };
    
    // Initial guess for the parameters, starting away from the minimum
    let initial_theta: Vec<f64> = vec![10.0; n_params];

    // --- Benchmarking ---
    // Create a new benchmark group
    let mut group = c.benchmark_group("ADAM Variants");

    // Benchmark the original implementation
    group.bench_function("adam_original_iterator", |b| {
        // The b.iter closure runs the code many times to get a measurement.
        b.iter(|| {
            // Use black_box to prevent the compiler from optimizing away
            // the function call or its parameters.
            adam_basic(
                black_box(alpha),
                black_box(beta1),
                black_box(beta2),
                black_box(eps),
                grad_fn, // Criterion can't black_box closures directly
                black_box(initial_theta.clone()),
                black_box(n_steps),
                black_box(convergence_tol),
            )
        })
    });

    // Benchmark the optimized implementation
    group.bench_function("adam_optimized_alpha_t", |b| {
        b.iter(|| {
            adam_optimized(
                black_box(alpha),
                black_box(beta1),
                black_box(beta2),
                black_box(eps),
                grad_fn,
                black_box(initial_theta.clone()),
                black_box(n_steps),
                black_box(convergence_tol),
            )
        })
    });

    group.finish();
}

// Boilerplate to register and run the benchmarks
criterion_group!(benches, run_adam_benchmarks);
criterion_main!(benches);
