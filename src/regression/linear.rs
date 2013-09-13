use std::vec;

use la::matrix::*;

use opt::graddescent;

struct LinearRegression {
  theta : ~Matrix<f64>,
  cost_history : ~[f64]
}

pub fn train(x : &Matrix<f64>, y : &Matrix<f64>, alpha : f64, num_iter : uint) -> LinearRegression {
  let extx = one_vector(x.rows()).cr(x);
  let mut theta = ~matrix(extx.cols(), 1, vec::from_elem(extx.cols(), 0.0f64));

  let dcost_cost_fn = |x : &Matrix<f64>, y : &Matrix<f64>, theta : &Matrix<f64>| -> (Matrix<f64>, f64) {
    // J(x) = 1/(2m) * SUM{i = 1 to m}: (theta . [1;x_i] - y_i)^2
    // dJ(x)/dtheta_j = 1/m * SUM{i = 1 to m}: (theta . [1;x_i] - y_i) * x_i_j

    let error = x * *theta - *y;
    let grad = (x.t() * error).scale(1.0f64 / (x.noRows as f64));
    let cost = (error.t() * error).scale(1.0 / (2.0 * (x.noRows as f64))).get(0, 0);
    (grad, cost)
  };

  let cost_history = graddescent::gradient_descent(&extx, y, theta, alpha, num_iter, dcost_cost_fn);
  LinearRegression {
    theta : theta,
    cost_history : cost_history
  } 
}

pub fn normal_eq(x : &Matrix<f64>, y : &Matrix<f64>) -> LinearRegression {
  let extx = one_vector(x.rows()).cr(x);
  LinearRegression {
    theta : ~((extx.t() * extx).inverse().unwrap() * extx.t() * *y),
    cost_history : ~[]
  }
}

impl LinearRegression {
  // h(x) = theta^T * [1; x]
  pub fn predict(&self, x : &Matrix<f64>) -> f64 {
    assert!(x.noCols == 1);
    assert!((x.noRows + 1) == self.theta.data.len());

    let mut sum = self.theta.data[0];
    for i in range(1, self.theta.data.len()) {
      sum += self.theta.data[i] * x.data[i - 1];
    }

    sum
  }

  #[inline]
  pub fn hypothesis(&self, x : &Matrix<f64>) -> f64 {
    self.predict(x)
  }
}

