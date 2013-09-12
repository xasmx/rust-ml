extern mod gnuplot;
extern mod la;
extern mod ml;

use std::vec;

use gnuplot::*;
use la::matrix::*;
use la::util::read_csv;
use ml::regression::linear;

fn main() {
  fn parser(s : &str) -> f64 { std::f64::from_str(s).unwrap() };
  let data = read_csv("example_data/linreg.csv", parser);

  let x = data.get_column(0);
  let y = data.get_column(1);

  let lr = linear::train(&x, &y, 0.005f64, 1000);

  let mut lx = vec::from_elem(2, 0.0f64);
  let mut ly = vec::from_elem(2, 0.0f64);
  for i in range(0, lx.len()) {
    lx[i] = (i as f64) * 30.0f64;
    ly[i] = lr.predict(&matrix(1, 1, ~[lx[i]]));
  }

  let mut fg = Figure::new();

  fg.axes2d()
  .set_aspect_ratio(Fix(1.0))
  .lines(lx.iter(), ly.iter(), [Color("#006633")])
  .points(x.data.iter(), y.data.iter(), [Caption("X"), PointSymbol('x'), Color("#ffaa77")])
  .set_title("Linear Regression", []);

  fg.show();
}

