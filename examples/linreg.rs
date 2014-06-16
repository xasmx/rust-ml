#![feature(globs)]

extern crate gnuplot;
extern crate la;
extern crate ml;

use std::from_str::FromStr;
use std::vec::Vec;

use gnuplot::*;
use la::Matrix;
use la::util::read_csv;
use ml::LinearRegression;
use ml::graph;

fn main() {
  fn parser(s : &str) -> f64 { FromStr::from_str(s).unwrap() };
  let data = read_csv("example_data/linreg.csv", parser);

  let x = data.get_column(0);
  let y = data.get_column(1);
  let mut cost_history = vec![];

  let lr = LinearRegression::train(&x, &y, 0.005f64, 100, Some(|cost| { cost_history.push(cost); }));

  let mut lx = Vec::from_elem(2, 0.0f64);
  let mut ly = Vec::from_elem(2, 0.0f64);
  for i in range(0, lx.len()) {
    *lx.get_mut(i) = (i as f64) * 30.0f64;
    *ly.get_mut(i) = lr.predict(&Matrix::new(1, 1, vec![*lx.get(i)]));
  }

  let mut fg = Figure::new();
  graph::draw_cost_graph(&mut fg, &cost_history);
  fg.show();

  let mut fg = Figure::new();

  fg.axes2d()
  .set_aspect_ratio(Fix(1.0))
  .lines(lx.iter(), ly.iter(), [Color("#006633")])
  .points(x.data.iter(), y.data.iter(), [PointSymbol('x'), Color("#ffaa77")])
  .set_title("Linear Regression", []);

  fg.show();
}

