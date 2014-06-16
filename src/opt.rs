use la::matrix::*;

// Gradient descent
pub fn gradient_descent(
    x : &Matrix<f64>,
    y : &Matrix<f64>,
    theta : &mut Matrix<f64>,
    alpha : f64,
    num_iter : uint,
    grad_f : |x : &Matrix<f64>, y : &Matrix<f64>, theta : &Matrix<f64>| -> Matrix<f64>)
{
  for _ in range(0, num_iter) {
    let grad = grad_f(x, y, theta);
    let step = grad.scale(alpha);
    theta.msub(&step);
  }
}
