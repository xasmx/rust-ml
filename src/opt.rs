use la::Matrix;

// Gradient descent
pub fn gradient_descent<F>(
    x : &Matrix<f64>,
    y : &Matrix<f64>,
    theta : &mut Matrix<f64>,
    alpha : f64,
    num_iter : usize,
    mut grad_f : F)
where F: FnMut(&Matrix<f64>, &Matrix<f64>, &Matrix<f64>) -> Matrix<f64>
{
  for _ in 0..num_iter {
    let grad = grad_f(x, y, theta);
    let step = grad.scale(alpha);
    theta.msub(&step);
  }
}
