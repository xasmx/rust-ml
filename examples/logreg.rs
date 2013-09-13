extern mod gnuplot;
extern mod la;
extern mod ml;

use std::vec;

use gnuplot::*;
use la::matrix::*;
use la::util::read_csv;
use ml::classification::logistic;
use ml::graph::costgraph;

fn main() {
  fn parser(s : &str) -> f64 { std::f64::from_str(s).unwrap() };
  let data = read_csv("example_data/logreg.csv", parser);

  let x = data.permute_columns([0, 1]);
  let y = data.get_column(2).map(|v : &f64| { if(*v == 0.0f64) { false } else { true } });

  let lr = logistic::train(&x, &y, 0.005f64, 100);

  let true_elements = data.select_rows(y.data);
  let true_x = true_elements.get_column(0);
  let true_y = true_elements.get_column(1);
  let false_elements = data.select_rows(y.map(|b| { !b }).data);
  let false_x = false_elements.get_column(0);
  let false_y = false_elements.get_column(1);

  //costgraph::show_cost_graph(lr.cost_history);

  let mut fg = Figure::new();

  fg.axes2d()
  .set_aspect_ratio(Fix(1.0))
  .points(true_x.data.iter(), true_y.data.iter(), [PointSymbol('x'), Color("#ffaa77")])
  .points(false_x.data.iter(), false_y.data.iter(), [PointSymbol('o'), Color("#333377")])
  .set_title("Logistic Regression", []);

  fg.show();
}

