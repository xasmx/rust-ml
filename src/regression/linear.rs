use std::vec;

use la::matrix::*;

use super::super::opt::graddescent;

struct LinearRegression {
  theta : ~Matrix<f64>
}

pub fn train(x : &Matrix<f64>, y : &Matrix<f64>, alpha : f64, num_iter : int) -> LinearRegression {
  let extx = one_vector(x.rows()).cr(x);
  let mut theta = ~matrix(extx.cols(), 1, vec::from_elem(extx.cols(), 0.0f64));
  graddescent::gradient_descent(&extx, y, theta, alpha, num_iter);
  LinearRegression {
    theta : theta
  } 
}

pub fn normal_eq(x : &Matrix<f64>, y : &Matrix<f64>) -> LinearRegression {
  let extx = one_vector(x.rows()).cr(x);
  LinearRegression {
    theta : ~((extx.t() * extx).inverse().unwrap() * extx.t() * *y)
  }
}

impl LinearRegression {
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

