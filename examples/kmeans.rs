extern crate gnuplot;
extern crate la;
extern crate ml;
extern crate rand;

use gnuplot::*;
use std::vec::Vec;

use la::Matrix;
use ml::KMeans;

fn main() {
  let mut v = vec![0.0f64; 200];
  for i in 0..100 {
    v[i] = 10.0 * rand::random::<f64>();
  }
  for i in 101..200 {
    v[i] = 10.0 + 10.0 * rand::random::<f64>();
  }
  
  let m = Matrix::new(100, 2, v);
  let kmeans = KMeans::cluster(2, &m);
  let assignments = kmeans.get_assignments();

  let mut xs = [Vec::with_capacity(100), Vec::with_capacity(100)];
  let mut ys = [Vec::with_capacity(100), Vec::with_capacity(100)];
  for i in 0..assignments.len() {
    xs[assignments[i]].push(m.get_data()[i * m.cols()]);
    ys[assignments[i]].push(m.get_data()[i * m.cols() + 1]);
  }

  let mut fg = Figure::new();
  
  fg.axes2d()
    .set_aspect_ratio(Fix(1.0))
    .points(xs[0].iter(), ys[0].iter(), &[Caption("X"), PointSymbol('x'), Color("#ffaa77")])
    .points(xs[1].iter(), ys[1].iter(), &[Caption("O"), PointSymbol('o'), Color("#ffaa77")])
    .set_title("K-Means", &[]);

  fg.show();
}

