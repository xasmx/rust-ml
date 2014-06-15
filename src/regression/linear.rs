use std::vec::Vec;

use la::matrix::*;

use opt::graddescent;

pub struct LinearRegression {
  theta : Matrix<f64>
}

pub fn train(x : &Matrix<f64>, y : &Matrix<f64>, alpha : f64, num_iter : uint, iter_notify_f_opt : Option<|f64| -> ()>) -> LinearRegression {
  let extx = one_vector(x.rows()).cr(x);
  let mut theta = matrix(extx.cols(), 1, Vec::from_elem(extx.cols(), 0.0f64));

  let calc_cost = iter_notify_f_opt.is_some();
  let f = if calc_cost { iter_notify_f_opt.unwrap() } else { |_| { } };
  let grad_f = |x : &Matrix<f64>, y : &Matrix<f64>, theta : &Matrix<f64>| -> Matrix<f64> {
    // J(x) = 1/(2m) * SUM{i = 1 to m}: (theta . [1;x_i] - y_i)^2
    // dJ(x)/dtheta_j = 1/m * SUM{i = 1 to m}: (theta . [1;x_i] - y_i) * x_i_j

    let error = x * *theta - *y;
    let grad = (x.t() * error).scale(1.0f64 / (x.rows() as f64));
    if calc_cost {
      let cost = (error.t() * error).scale(1.0 / (2.0 * (x.rows() as f64))).get(0, 0);
      f(cost);
    }
    grad
  };

  graddescent::gradient_descent(&extx, y, &mut theta, alpha, num_iter, grad_f);
  LinearRegression {
    theta : theta
  } 
}

pub fn normal_eq(x : &Matrix<f64>, y : &Matrix<f64>) -> LinearRegression {
  let extx = one_vector(x.rows()).cr(x);
  LinearRegression {
    theta : (extx.t() * extx).inverse().unwrap() * extx.t() * *y
  }
}

impl LinearRegression {
  // h(x) = theta^T * [1; x]
  pub fn predict(&self, x : &Matrix<f64>) -> f64 {
    assert!(x.cols() == 1);
    assert!((x.rows() + 1) == self.theta.data.len());

    let mut sum = *self.theta.data.get(0);
    for i in range(1, self.theta.data.len()) {
      sum += *self.theta.data.get(i) * *x.data.get(i - 1);
    }

    sum
  }

  #[inline]
  pub fn hypothesis(&self, x : &Matrix<f64>) -> f64 {
    self.predict(x)
  }
}

