use la::Matrix;

pub struct GradientDescent<'a> {
  learning_rate : f64,
  learnable_weights : &'a mut Matrix<f64>,
  error_f : &'a Fn(&Matrix<f64>) -> Matrix<f64>,
  grad_f : &'a Fn(&Matrix<f64>) -> Matrix<f64>
}

pub struct IterationResult {
  pub error : Matrix<f64>,
  pub grad : Matrix<f64>
}

impl <'a> GradientDescent<'a> {
  pub fn new(
      learning_rate : f64,
      learnable_weights : &'a mut Matrix<f64>,
      error_f : &'a Fn(&Matrix<f64>) -> Matrix<f64>,
      grad_f : &'a Fn(&Matrix<f64>) -> Matrix<f64>) -> GradientDescent<'a> {
    GradientDescent {
      learning_rate : learning_rate,
      learnable_weights : learnable_weights,
      error_f : error_f,
      grad_f : grad_f
    }
  }

  pub fn iterate(&mut self) -> IterationResult {
    let error = (self.error_f)(self.learnable_weights);
    let mut grad = (self.grad_f)(&error);
    grad.mscale(self.learning_rate);
    self.learnable_weights.msub(&grad);
    IterationResult {
      error : error,
      grad : grad
    }
  }
}

