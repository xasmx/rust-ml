#[feature(globs)];

extern mod gnuplot;
extern mod la;
extern mod ml;

use std::rand;
use std::vec;

use gnuplot::*;
use la::matrix::*;
use ml::clustering::kmeans;

fn main() {
  let mut v = vec::from_elem(200, 0.0f64);
  for i in range(0, 100) {
    v[i] = 10.0 * rand::random();
  }
  for i in range(101, 200) {
    v[i] = 10.0 + 10.0 * rand::random();
  }
  
  let m = matrix(100, 2, v);
  let assignments = kmeans::kmeans(2, &m);

  let mut xs = [vec::with_capacity(100), vec::with_capacity(100)];
  let mut ys = [vec::with_capacity(100), vec::with_capacity(100)];
  for i in range(0, assignments.len()) {
    xs[assignments[i]].push(m.data[i * m.cols()]);
    ys[assignments[i]].push(m.data[i * m.cols() + 1]);
  }

  let mut fg = Figure::new();
  
  fg.axes2d()
  .set_aspect_ratio(Fix(1.0))
  .points(xs[0].iter(), ys[0].iter(), [Caption("X"), PointSymbol('x'), Color("#ffaa77")])
  .points(xs[1].iter(), ys[1].iter(), [Caption("O"), PointSymbol('o'), Color("#ffaa77")])
  .set_title("K-Means", []);

  fg.show();
}

