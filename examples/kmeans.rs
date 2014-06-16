#![feature(globs)]

extern crate gnuplot;
extern crate la;
extern crate ml;

use std::rand;
use std::vec::Vec;

use gnuplot::*;
use la::matrix::*;
use ml::KMeans;

fn main() {
  let mut v = Vec::from_elem(200, 0.0f64);
  for i in range(0, 100) {
    *v.get_mut(i as uint) = 10.0 * rand::random();
  }
  for i in range(101, 200) {
    *v.get_mut(i as uint) = 10.0 + 10.0 * rand::random();
  }
  
  let m = matrix(100, 2, v);
  let kmeans = KMeans::cluster(2, &m);
  let assignments = kmeans.get_assignments();

  let mut xs = [Vec::with_capacity(100), Vec::with_capacity(100)];
  let mut ys = [Vec::with_capacity(100), Vec::with_capacity(100)];
  for i in range(0, assignments.len()) {
    xs[*assignments.get(i)].push(*m.data.get(i * m.cols()));
    ys[*assignments.get(i)].push(*m.data.get(i * m.cols() + 1));
  }

  let mut fg = Figure::new();
  
  fg.axes2d()
  .set_aspect_ratio(Fix(1.0))
  .points(xs[0].iter(), ys[0].iter(), [Caption("X"), PointSymbol('x'), Color("#ffaa77")])
  .points(xs[1].iter(), ys[1].iter(), [Caption("O"), PointSymbol('o'), Color("#ffaa77")])
  .set_title("K-Means", []);

  fg.show();
}

