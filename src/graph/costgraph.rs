use std::vec;

use gnuplot::*;

pub fn show_cost_graph(cost_history : &[f64]) {
  let mut x = vec::with_capacity(cost_history.len());
  for i in range(0, cost_history.len()) {
    x.push(i)
  }

  let mut fg = Figure::new();

  fg.axes2d()
  .set_aspect_ratio(Fix(1.0))
  .set_x_label("Iteration", [Rotate(0.0)])
  .set_y_label("Cost", [Rotate(90.0)])
  .lines(x.iter(), cost_history.iter(), [Color("#006633")])
  .set_title("Cost Graph", []);

  fg.show();
}


