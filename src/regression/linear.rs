use la::Matrix;

use opt::GradientDescent;

pub struct LinearRegressionBuilder<'a> {
  learning_rate : f64,
  max_iterations : usize,
  cost_history_notify_f : Option<&'a mut FnMut(f64)>
}

pub struct LinearRegression {
  theta : Matrix<f64>
}

impl <'a> LinearRegressionBuilder<'a> {
  pub fn learning_rate(mut self, learning_rate : f64) -> LinearRegressionBuilder<'a> {
    self.learning_rate = learning_rate;
    self
  }

  pub fn max_iterations(mut self, max_iterations : usize) -> LinearRegressionBuilder<'a> {
    self.max_iterations = max_iterations;
    self
  }

  pub fn cost_history(mut self, cost_notify_f : &'a mut FnMut(f64)) -> LinearRegressionBuilder<'a> {
    self.cost_history_notify_f = Some(cost_notify_f);
    self
  }

  pub fn train(&mut self, x : &Matrix<f64>, y : &Matrix<f64>) -> LinearRegression {
    let learning_rate = self.learning_rate;
    let max_iterations = self.max_iterations;
    let extx = Matrix::one_vector(x.rows()).cr(x);
    let mut theta = Matrix::new(extx.cols(), 1, vec![0.0f64; extx.cols()]);
    let error_f = |theta : &Matrix<f64>| -> Matrix<f64> {
      &extx * theta - y
    };
    let cost_f = |error : &Matrix<f64>| -> f64 {
      // J(x) = 1/(2m) * SUM{i = 1 to m}: (theta . [1;x_i] - y_i)^2
      error.dot(&error) / (2.0 * (x.rows() as f64))
    };
    let grad_f = |error : &Matrix<f64>| -> Matrix<f64> {
      // dJ(x)/dtheta_j = 1/m * SUM{i = 1 to m}: (theta . [1;x_i] - y_i) * x_i_j
      (extx.t() * error).scale(1.0f64 / extx.rows() as f64)
    };

    {
      let mut grad_desc = GradientDescent::new(learning_rate, &mut theta, &error_f, &grad_f);
      for _ in 0..max_iterations {
        let res = grad_desc.iterate();
        match &mut self.cost_history_notify_f {
          &mut Some(ref mut f) => {
            f(cost_f(&res.error));
          }
          _ => { }
        }
      }
    }

    LinearRegression {
      theta : theta
    } 
  }
}

impl LinearRegression {
  pub fn new<'a>() -> LinearRegressionBuilder<'a> {
    LinearRegressionBuilder {
      learning_rate : 0.005f64,
      max_iterations : 100,
      cost_history_notify_f : None
    }
  }

  pub fn normal_eq(x : &Matrix<f64>, y : &Matrix<f64>) -> LinearRegression {
    let extx = Matrix::one_vector(x.rows()).cr(x);
    LinearRegression {
      theta : (extx.t() * &extx).inverse().unwrap() * extx.t() * y
    }
  }

  // h(x) = theta^T * [1; x]
  pub fn predict(&self, x : &Matrix<f64>) -> f64 {
    assert!(x.cols() == 1);
    assert!((x.rows() + 1) == self.theta.get_data().len());

    let mut sum = self.theta.get_data()[0];
    for i in 1..self.theta.get_data().len() {
      sum += self.theta.get_data()[i] * x.get_data()[i - 1];
    }

    sum
  }

  #[inline]
  pub fn hypothesis(&self, x : &Matrix<f64>) -> f64 {
    self.predict(x)
  }
}

