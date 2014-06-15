use std::vec::Vec;

use gnuplot::*;

use classification::logistic::LogisticRegression;
use la::matrix::*;

pub fn show_decision_boundary_2d(fg : &mut Figure, lr : &LogisticRegression) {
  let mut z_array : Vec<f64> = vec![];
  let no_cols = 40u;
  let x_start = 30.0f64;
  let x_end = 100.0f64;
  let x_step = (x_end - x_start) / no_cols as f64;
  let no_rows = 40u;
  let y_start = 30.0f64;
  let y_end = 100.0f64;
  let y_step = (y_end - y_start) / no_rows as f64;

  for row in range(0, no_rows) {
    for col in range(0, no_cols) {
      let x = vector(vec![x_start + col as f64 * x_step, y_start + row as f64 * y_step]);
      let p = lr.p(&x) - lr.get_threshold();
      z_array.push(p);
    }
  }

  fg.axes3d()
  .surface(z_array.iter(), no_rows, no_cols, Some((x_start, y_start, x_end, y_end)), &[])
  .show_contours_custom(true, true, Linear, Fix("Decision Boundary"), [0.0f64].iter());
}

