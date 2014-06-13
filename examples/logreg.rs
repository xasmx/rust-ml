#![feature(globs)]

extern crate gnuplot;
extern crate la;
extern crate ml;

use std::from_str::FromStr;

use gnuplot::*;
use la::util::read_csv;
use ml::classification::logistic;
use ml::graph::{costgraph,decisionboundary};

fn main() {
  fn parser(s : &str) -> f64 { FromStr::from_str(s).unwrap() };
  let data = read_csv("example_data/logreg.csv", parser);

  let x = data.permute_columns(&vec![0, 1]);
  let y = data.get_column(2).map(|v : &f64| { if *v == 0.0f64 { false } else { true } });
  let lr = logistic::train(&x, &y, 0.005f64, 10000);

  let true_elements = data.select_rows(y.data.as_slice());
  let true_x = true_elements.get_column(0);
  let true_y = true_elements.get_column(1);
  let false_elements = data.select_rows(y.map(|b| { !b }).data.as_slice());
  let false_x = false_elements.get_column(0);
  let false_y = false_elements.get_column(1);

  costgraph::show_cost_graph(&lr.cost_history);
  //decisionboundary::show_decision_boundary_2d(&lr);

/*
for i in range(0u, 100u) {
  //let p = lr.p(&vector(~[false_x.get(i, 0), false_y.get(i, 0)]));
  //io::println(fmt!("%?: %?", lr.predict(&vector(~[false_x.get(i, 0), false_y.get(i, 0)])), p));
  let p = lr.p(&vector(~[true_x.get(i, 0), true_y.get(i, 0)]));
  io::println(fmt!("%?: %?", lr.predict(&vector(~[true_x.get(i, 0), true_y.get(i, 0)])), p));
}
*/

  let mut fg = Figure::new();

  fg.axes2d()
  .set_aspect_ratio(Fix(1.0))
  .points(true_x.data.iter(), true_y.data.iter(), [PointSymbol('x'), Color("#ffaa77")])
  .points(false_x.data.iter(), false_y.data.iter(), [PointSymbol('o'), Color("#333377")])
  .set_title("Logistic Regression", []);

  fg.show();
}

