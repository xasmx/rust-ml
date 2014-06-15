#![feature(globs)]

extern crate gnuplot;
extern crate la;
extern crate ml;

use std::from_str::FromStr;
use std::vec::Vec;

use gnuplot::*;
use la::matrix::*;
use la::util::read_csv;
use ml::regression::linear;
use ml::graph::costgraph;

fn main() {
  fn parser(s : &str) -> f64 { FromStr::from_str(s).unwrap() };
  let data = read_csv("example_data/linreg.csv", parser);

  let x = data.get_column(0);
  let y = data.get_column(1);

  let lr = linear::train(&x, &y, 0.005f64, 100);

  let mut lx = Vec::from_elem(2, 0.0f64);
  let mut ly = Vec::from_elem(2, 0.0f64);
  for i in range(0, lx.len()) {
    *lx.get_mut(i) = (i as f64) * 30.0f64;
    *ly.get_mut(i) = lr.predict(&matrix(1, 1, vec![*lx.get(i)]));
  }

  let mut fg = Figure::new();
  costgraph::show_cost_graph(&mut fg, &lr.cost_history);
  fg.show();

  let mut fg = Figure::new();

  fg.axes2d()
  .set_aspect_ratio(Fix(1.0))
  .lines(lx.iter(), ly.iter(), [Color("#006633")])
  .points(x.data.iter(), y.data.iter(), [PointSymbol('x'), Color("#ffaa77")])
  .set_title("Linear Regression", []);

  fg.show();
}

