#[allow(clippy::too_many_arguments)]
pub fn adam_basic(
    alpha: f64, // stepsize
    beta1: f64, // exponential decay rate for first moment estimate
    beta2: f64, // exponential decay rate for second moment estimate
    eps: f64, // epsilon tolerance
    grad_fn: impl Fn(&[f64]) -> Vec<f64>, // the gradient of the function differentiable wrt its parameters theta
    mut theta: Vec<f64>, // the function parameters
    n_steps: i32, // maximum number of steps before we break regardless of convergence
    convergence_tol: f64,
) -> Vec<f64> {

    assert!((0.0..1.0).contains(&beta1), "beta1 must be in the range [0, 1)");
    assert!((0.0..1.0).contains(&beta2), "beta2 must be in the range [0, 1)");

    let n_params = theta.len();

    // initialize first moment vector
    let mut m_t: Vec<f64> = vec![0.0; n_params];
    // initialize second moment vector
    let mut v_t: Vec<f64> = vec![0.0; n_params];

    // ie while we have not converged
    for t in 1..=n_steps {
        let g_t: Vec<f64> = grad_fn(&theta);

        m_t = m_t.iter().zip(g_t.iter()).map(|(m, g)| {
            beta1 * m + (1.0 - beta1) * g
        }).collect();

        v_t = v_t.iter().zip(g_t.iter()).map(|(v, g)| {
            beta2 * v + (1.0 - beta2) * g.powi(2)
        }).collect();

        let m_t_est: Vec<f64> = m_t.iter().map(|m| {
            m / (1.0 - beta1.powi(t))
        }).collect();

        let v_t_est: Vec<f64> = v_t.iter().map(|v| {
            v / (1.0 - beta2.powi(t))
        }).collect();

        let theta_old = theta.clone();

        theta = theta.iter().zip(m_t_est).zip(v_t_est).map(|((t, m), v)| {
            t - (alpha * m / (v.sqrt() + eps))
        }).collect();

        let distance_sq: f64 = theta_old.iter()
            .zip(theta.iter())
            .map(|(old, new)| (new - old).powi(2))
            .sum();

        if distance_sq < convergence_tol.powi(2) {
            // println!("Converged after {} steps", t);
            break;
        }
    }
    theta
}


#[allow(clippy::too_many_arguments)]
pub fn adam_optimized(
    alpha: f64, // stepsize
    beta1: f64, // exponential decay rate for first moment estimate
    beta2: f64, // exponential decay rate for second moment estimate
    eps: f64, // epsilon tolerance
    grad_fn: impl Fn(&[f64]) -> Vec<f64>, // the gradient of the function differentiable wrt its parameters theta
    mut theta: Vec<f64>, // the function parameters
    n_steps: i32, // maximum number of steps before we break regardless of convergence
    convergence_tol: f64,
) -> Vec<f64> {

    assert!((0.0..1.0).contains(&beta1), "beta1 must be in the range [0, 1)");
    assert!((0.0..1.0).contains(&beta2), "beta2 must be in the range [0, 1)");

    let n_params = theta.len();

    // initialize first moment vector
    let mut m_t: Vec<f64> = vec![0.0; n_params];
    // initialize second moment vector
    let mut v_t: Vec<f64> = vec![0.0; n_params];

    // ie while we have not converged
    for t in 1..=n_steps {
        let g_t: Vec<f64> = grad_fn(&theta);

        m_t = m_t.iter().zip(g_t.iter()).map(|(m, g)| {
            beta1 * m + (1.0 - beta1) * g
        }).collect();

        v_t = v_t.iter().zip(g_t.iter()).map(|(v, g)| {
            beta2 * v + (1.0 - beta2) * g.powi(2)
        }).collect();

        let alpha_t = alpha * (1.0 - beta2.powi(t)).sqrt() / (1.0 - beta1.powi(t));

        let theta_old = theta.clone();

        theta = theta.iter().zip(m_t.iter()).zip(v_t.iter()).map(|((t, m), v)| {
            t - alpha_t * m / (v.sqrt() + eps)
        }).collect();

        let distance_sq: f64 = theta_old.iter()
            .zip(theta.iter())
            .map(|(old, new)| (new - old).powi(2))
            .sum();

        if distance_sq < convergence_tol.powi(2) {
            // println!("Converged after {} steps", t);
            break;
        }
    }
    theta
}
