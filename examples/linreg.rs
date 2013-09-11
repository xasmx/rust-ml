extern mod gnuplot;
extern mod la;
extern mod ml;

use std::rand;
use std::vec;

use gnuplot::*;
use la::matrix::*;
use ml::regression::linear;

fn main() {
  let mut xdata = vec::from_elem(100, 0.0f64);
  for i in range(0, xdata.len()) {
    xdata[i] = 10.0 * rand::random();
  }
  let mut ydata = vec::from_elem(100, 0.0f64);
  for i in range(0, ydata.len()) {
    ydata[i] = 4.0 * rand::random();
  }
 
  let x = matrix(100, 1, xdata);
  let y = matrix(100, 1, ydata); 
  let lr = linear::train(&x, &y, 0.05f64, 1000);

  let mut lx = vec::from_elem(2, 0.0f64);
  let mut ly = vec::from_elem(2, 0.0f64);
  for i in range(0, lx.len()) {
    lx[i] = (i as f64) * 10.0f64;
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

