use la::matrix::*;

// Gradient descent
//
pub fn gradient_descent(x : &Matrix<f64>, y : &Matrix<f64>, theta : &mut Matrix<f64>, alpha : f64, num_iter : int) {
  // Cost function:
  // J(x) = 1/(2m) * SUM{i = 1 to m}: (theta . [1;x_i] - y_i)^2
  // dJ(x)/dtheta_0 = 1/m * SUM{i = 1 to m}: (theta . [1;x_i] - y_i)
  // dJ(x)/dtheta_1 = 1/m * SUM{i = 1 to m}: (theta . [1;x_i] - y_i) * x_i_1
  for _ in range(0, num_iter) {
    let error = x * *theta - *y;
    theta.msub(&(x.t() * error).scale(alpha / (x.noRows as f64)));

    //let cost = (error.t() * error).scale(1.0 / (2.0 * (x.noRows as f64))).get(0, 0);
    //println(fmt!("%?", cost));
  }
}


