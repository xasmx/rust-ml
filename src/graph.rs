use std::vec::Vec;

use gnuplot::*;

/// Draws a graph with x-axis indicating the iteration number and
/// y-axis showing the cost at that iteration.
pub fn draw_cost_graph(fg : &mut Figure, cost_history : &Vec<f64>) {
  let mut x = Vec::with_capacity(cost_history.len());
  for i in 0..cost_history.len() {
    x.push(i)
  }

  fg.axes2d()
    .set_aspect_ratio(Fix(1.0))
    .set_x_label("Iteration", &[Rotate(0.0)])
    .set_y_label("Cost", &[Rotate(90.0)])
    .lines(x.iter(), cost_history.iter(), &[Color("#006633")])
    .set_title("Cost Graph", &[]);
}

/// Draws a decision boundary at p_f(x, y) = 0 by evaluating the function
/// within the supplied limits. The function in evaluated as a grid of size
/// (grid_size x grid_size).
pub fn draw_decision_boundary_2d(fg : &mut Figure, limits : (f64, f64, f64, f64), grid_size : usize, p_f : &Fn(f64, f64) -> f64) {
  assert!(limits.0 < limits.2);
  assert!(limits.1 < limits.3);
  assert!(grid_size >= 1);

  let mut z_array : Vec<f64> = vec![];
  let x_start = limits.0;
  let y_start = limits.1;
  let x_end = limits.2;
  let y_end = limits.3;
  let x_step = (x_end - x_start) / grid_size as f64;
  let y_step = (y_end - y_start) / grid_size as f64;

  for row in 0..grid_size {
    for col in 0..grid_size {
      let z = p_f(x_start + col as f64 * x_step, y_start + row as f64 * y_step);
      z_array.push(z);
    }
  }

  fg.axes3d()
    .surface(z_array.iter(), grid_size, grid_size, Some((x_start, y_start, x_end, y_end)), &[])
    .show_contours_custom(true, true, Linear, Fix("Decision Boundary"), [0.0f64].iter());
}

