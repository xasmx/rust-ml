use std::num;
use std::vec;

use la::matrix::*;

use opt::graddescent;

struct LogisticRegression {
  theta : ~Matrix<f64>,
  threshold : f64,
  cost_history : ~[f64]
}

pub fn train(x : &Matrix<f64>, y : &Matrix<bool>, alpha : f64, num_iter : uint) -> LogisticRegression {
  let extx = one_vector(x.rows()).cr(x);
  let numy = y.map(|b : &bool| -> f64 { if *b { 1.0 } else { 0.0 } });
  let mut theta = ~matrix(extx.cols(), 1, vec::from_elem(extx.cols(), 0.0f64));

  let dcost_cost_fn = |x : &Matrix<f64>, y : &Matrix<f64>, theta : &Matrix<f64>| -> (Matrix<f64>, f64) {
    // J(theta) = 1 / m * SUM{i = 1 to m}: Cost(h_theta(x), y)
    // Cost(h_theta(x), y) = { - log(h_theta(x))		, if y = true (1)
    //                       , - log(1 - h_theta(x)) }		, if y = false (0)
    //                     = - y * log(h_theta(x)) - (1 - y) * log(1 - h_theta(x))
    // error = 1 / (1 + e^(- theta^T * x)) - y
    // grad = 1 / m * x' * error
    let error = (x * *theta).map(|t : &f64| -> f64 { 1.0f64 / (1.0f64 + num::exp(*t)) }) - *y;
    let grad = (x.t() * error).scale(1.0f64 / (x.noRows as f64));
    //let cost = (error.t() * error).scale(1.0 / (2.0 * (x.noRows as f64))).get(0, 0);
    (grad, 0.0f64)
  };

  let cost_history = graddescent::gradient_descent(&extx, &numy, theta, alpha, num_iter, dcost_cost_fn);
  LogisticRegression {
    theta : theta,
    threshold : 0.5,
    cost_history : cost_history
  } 
}

impl LogisticRegression {
  #[inline]
  pub fn set_threshold(&mut self, threshold : f64) {
    self.threshold = threshold;
  }

  // h(x) = 1 / (1 + e^(- theta^T * [1; x]))
  pub fn predict(&self, x : &Matrix<f64>) -> bool {
    assert!(x.noCols == 1);
    assert!((x.noRows + 1) == self.theta.data.len());

    // sum = theta^T * x
    let mut sum = self.theta.data[0];
    for i in range(1, self.theta.data.len()) {
      sum += self.theta.data[i] * x.data[i - 1];
    }

    let p = 1.0 / (1.0 + num::exp(- sum));
    (p >= self.threshold)
  }

  #[inline]
  pub fn hypothesis(&self, x : &Matrix<f64>) -> bool {
    self.predict(x)
  }
}

