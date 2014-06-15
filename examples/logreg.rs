#![feature(globs)]

extern crate gnuplot;
extern crate la;
extern crate ml;

use std::from_str::FromStr;

use gnuplot::*;
use la::matrix::*;
use la::util::read_csv;
use ml::classification::logistic;
use ml::graph::{costgraph,decisionboundary};

fn main() {
  fn parser(s : &str) -> f64 { FromStr::from_str(s).unwrap() };
  let data = read_csv("example_data/logreg.csv", parser);

  let x = data.permute_columns(&vec![0, 1]);
  let y = data.get_column(2).map(|v : &f64| { if *v == 0.0f64 { false } else { true } });
  let mut cost_history = Vec::new();
  let mut iter_count = 0;
  let lr = logistic::train(&x, &y, 0.001f64, 500000, &mut Some(|cost| { if iter_count % 1000 == 0 { cost_history.push(cost); } iter_count += 1; }));

  let true_elements = data.select_rows(y.data.as_slice());
  let true_x = true_elements.get_column(0);
  let true_y = true_elements.get_column(1);
  let false_elements = data.select_rows(y.map(|b| { !b }).data.as_slice());
  let false_x = false_elements.get_column(0);
  let false_y = false_elements.get_column(1);

  let mut fg = Figure::new();
  costgraph::show_cost_graph(&mut fg, &cost_history);
  fg.show();

  let mut fg = Figure::new();
  decisionboundary::show_decision_boundary_2d(&mut fg, &lr);
  fg.show();

  println!("falses:");
  for i in range(0u, false_x.data.len()) {
    let p = lr.p(&vector(vec![false_x.get(i, 0), false_y.get(i, 0)]));
    println!("  {}: {}", lr.predict(&vector(vec![false_x.get(i, 0), false_y.get(i, 0)])), p);
  }
  println!("trues:");
  for i in range(0u, true_x.data.len()) {
    let p = lr.p(&vector(vec![true_x.get(i, 0), true_y.get(i, 0)]));
    println!("  {}: {}", lr.predict(&vector(vec![true_x.get(i, 0), true_y.get(i, 0)])), p);
  }

  // Decision boundary is at t'x = 0
  //   x_1 = - (t_0 + t_1 * x_0) / t_2
  let theta = lr.get_theta();
  let lx = vec![30.0f64, 100.0f64];
  let y0 = - (theta.get(0, 0) + theta.get(1, 0) * *lx.get(0)) / theta.get(2, 0);
  let y1 = - (theta.get(0, 0) + theta.get(1, 0) * *lx.get(1)) / theta.get(2, 0);
  let ly = vec![y0, y1];

  let mut fg = Figure::new();

  fg.axes2d()
  .set_aspect_ratio(Fix(1.0))
  .points(true_x.data.iter(), true_y.data.iter(), [PointSymbol('x'), Color("#ffaa77")])
  .points(false_x.data.iter(), false_y.data.iter(), [PointSymbol('o'), Color("#333377")])
  .lines(lx.iter(), ly.iter(), [Color("#006633")])
  .set_title("Logistic Regression", []);

  fg.show();
}

