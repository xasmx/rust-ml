use std::vec::Vec;

use la::Matrix;
use opt;

pub struct LogisticRegression {
  theta : Matrix<f64>,
  threshold : f64
}

impl LogisticRegression {
  pub fn train(x : &Matrix<f64>, y : &Matrix<bool>, alpha : f64, num_iter : uint, iter_notify_f_opt : Option<|f64| -> ()>) -> LogisticRegression {
    let extx = Matrix::one_vector(x.rows()).cr(x);
    let numy = y.map(|b : &bool| -> f64 { if *b { 1.0 } else { 0.0 } });
    let mut theta = Matrix::new(extx.cols(), 1, Vec::from_elem(extx.cols(), 0.0f64));

    let calc_cost = iter_notify_f_opt.is_some();
    let f = if calc_cost { iter_notify_f_opt.unwrap() } else { |_| { } };
    let grad_f = |x : &Matrix<f64>, y : &Matrix<f64>, theta : &Matrix<f64>| -> Matrix<f64> {
      // J(theta) = 1 / m * SUM{i = 1 to m}: Cost(h_theta(x), y)
      // Cost(h_theta(x), y) = { - log(h_theta(x))		, if y = true (1)
      //                       , - log(1 - h_theta(x)) }	, if y = false (0)
      //                     = - y * log(h_theta(x)) - (1 - y) * log(1 - h_theta(x))
      // error = 1 / (1 + e^(- theta^T * x)) - y
      // grad = 1 / m * x' * error
      let error = (x * *theta).map(|t : &f64| -> f64 { 1.0f64 / (1.0f64 + (- *t).exp()) }) - *y;
      let grad = (x.t() * error).scale(1.0f64 / (x.rows() as f64));
      if calc_cost {
        let cost = (error.t() * error).scale(1.0 / x.rows() as f64).get(0, 0);
        f(cost);
      }
      grad
    };

    opt::gradient_descent(&extx, &numy, &mut theta, alpha, num_iter, grad_f);
    LogisticRegression {
      theta : theta,
      threshold : 0.5
    } 
  }

  pub fn get_theta<'a>(&'a self) -> &'a Matrix<f64> {
    &self.theta
  }

  pub fn get_threshold(&self) -> f64 {
    self.threshold
  }

  #[inline]
  pub fn set_threshold(&mut self, threshold : f64) {
    self.threshold = threshold;
  }

  // h(x) = 1 / (1 + e^(- theta^T * [1; x]))
  pub fn p(&self, x : &Matrix<f64>) -> f64 {
    assert!(x.cols() == 1);
    assert!((x.rows() + 1) == self.theta.data.len());

    // sum = theta^T * x
    let mut sum = *self.theta.data.get(0);
    for i in range(1, self.theta.data.len()) {
      sum += *self.theta.data.get(i) * *x.data.get(i - 1);
    }

    1.0 / (1.0 + (- sum).exp())
  }

  // h(x) >= threshold
  pub fn predict(&self, x : &Matrix<f64>) -> bool {
    (self.p(x) >= self.threshold)
  }

  #[inline]
  pub fn hypothesis(&self, x : &Matrix<f64>) -> bool {
    self.predict(x)
  }
}

