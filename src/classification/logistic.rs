use la::Matrix;
use opt::GradientDescent;

pub struct LogisticRegression {
  theta : Matrix<f64>,
  threshold : f64
}

impl LogisticRegression {
  pub fn train<F : FnMut(f64) -> ()>(x : &Matrix<f64>, y : &Matrix<bool>, learning_rate : f64, max_iterations : usize, mut iter_notify_f_opt : Option<F>) -> LogisticRegression {
    let extx = Matrix::one_vector(x.rows()).cr(x);
    let numy = y.map(&|b : &bool| -> f64 { if *b { 1.0 } else { 0.0 } });
    let mut theta = Matrix::new(extx.cols(), 1, vec![0.0f64; extx.cols()]);

    let error_f = |theta : &Matrix<f64>| {
      (&extx * theta).map(&|t : &f64| -> f64 { 1.0f64 / (1.0f64 + (- *t).exp()) }) - &numy
    };
    let grad_f = |error : &Matrix<f64>| {
      // error = 1 / (1 + e^(- theta^T * x)) - y
      // grad = 1 / m * x' * error
      (extx.t() * error).scale(1.0f64 / extx.rows() as f64)
    };
    let cost_f = |error : &Matrix<f64>| {
      // J(theta) = 1 / m * SUM{i = 1 to m}: Cost(h_theta(x), y)
      // Cost(h_theta(x), y) = { - log(h_theta(x))		, if y = true (1)
      //                       , - log(1 - h_theta(x)) }	, if y = false (0)
      //                     = - y * log(h_theta(x)) - (1 - y) * log(1 - h_theta(x))
      error.dot(&error) / extx.rows() as f64
    };

    {
      let mut grad_desc = GradientDescent::new(learning_rate, &mut theta, &error_f, &grad_f);
      for _ in 0..max_iterations {
        let res = grad_desc.iterate();
        match &mut iter_notify_f_opt {
          &mut Some(ref mut f) => {
            f(cost_f(&res.error));
          }
          _ => { }
        }
      }
    }

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
    assert!((x.rows() + 1) == self.theta.get_data().len());

    // sum = theta^T * x
    let mut sum = self.theta.get_data()[0];
    for i in 1..self.theta.get_data().len() {
      sum += self.theta.get_data()[i] * x.get_data()[i - 1];
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

