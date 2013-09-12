use std::vec;

use la::matrix::*;

// Gradient descent
pub fn gradient_descent(
    x : &Matrix<f64>,
    y : &Matrix<f64>,
    theta : &mut Matrix<f64>,
    alpha : f64,
    num_iter : uint,
    dcost_cost_fn : &fn(x : &Matrix<f64>, y : &Matrix<f64>, theta : &Matrix<f64>) -> (Matrix<f64>, f64))
-> ~[f64]
{
  let mut cost_history = vec::with_capacity(num_iter);
  for _ in range(0, num_iter) {
    let (grad, cost) = dcost_cost_fn(x, y, theta);
    let step = grad.scale(alpha);
    theta.msub(&step);
    cost_history.push(cost);
  }

  cost_history
}
