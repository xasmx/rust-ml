extern crate gnuplot;
extern crate la;
extern crate ml;

use std::str::FromStr;

use gnuplot::*;
use la::Matrix;
use la::util::read_csv;
use ml::LogisticRegression;
use ml::graph;


fn main() {
  fn parser(s : &str) -> f64 { FromStr::from_str(s).unwrap() };
  let data = read_csv("example_data/logreg.csv", &parser);

  let x = data.permute_columns(&vec![0, 1]);
  let y = data.get_column(2).map(&|v : &f64| { if *v == 0.0f64 { false } else { true } });
  let mut cost_history = Vec::new();
  let mut iter_count = 0;
  let lr = LogisticRegression::train(&x, &y, 0.001f64, 500000, Some(|cost| {
    if iter_count % 1000 == 0 { cost_history.push(cost); } iter_count += 1;
  }));

  let true_elements = data.select_rows(&y.get_data());
  let true_x = true_elements.get_column(0);
  let true_y = true_elements.get_column(1);
  let false_elements = data.select_rows(&y.map(&|b| { !b }).get_data());
  let false_x = false_elements.get_column(0);
  let false_y = false_elements.get_column(1);

  let mut fg = Figure::new();
  graph::draw_cost_graph(&mut fg, &cost_history);
  fg.show();

  let mut fg = Figure::new();
  graph::draw_decision_boundary_2d(&mut fg, (30.0, 30.0, 100.0, 100.0), 50, &|x, y| {
    lr.p(&Matrix::vector(vec![x, y])) - lr.get_threshold()
  });
  fg.show();

  println!("falses:");
  for i in 0..false_x.get_data().len() {
    let p = lr.p(&Matrix::vector(vec![false_x.get(i, 0), false_y.get(i, 0)]));
    println!("  {}: {}", lr.predict(&Matrix::vector(vec![false_x.get(i, 0), false_y.get(i, 0)])), p);
  }
  println!("trues:");
  for i in 0..true_x.get_data().len() {
    let p = lr.p(&Matrix::vector(vec![true_x.get(i, 0), true_y.get(i, 0)]));
    println!("  {}: {}", lr.predict(&Matrix::vector(vec![true_x.get(i, 0), true_y.get(i, 0)])), p);
  }

  // Decision boundary is at t'x = 0
  //   x_1 = - (t_0 + t_1 * x_0) / t_2
  let theta = lr.get_theta();
  let lx = vec![30.0f64, 100.0f64];
  let y0 = - (theta.get(0, 0) + theta.get(1, 0) * lx[0]) / theta.get(2, 0);
  let y1 = - (theta.get(0, 0) + theta.get(1, 0) * lx[1]) / theta.get(2, 0);
  let ly = vec![y0, y1];

  let mut fg = Figure::new();

  fg.axes2d()
  .set_aspect_ratio(Fix(1.0))
  .points(true_x.get_data().iter(), true_y.get_data().iter(), &[PointSymbol('x'), Color("#ffaa77")])
  .points(false_x.get_data().iter(), false_y.get_data().iter(), &[PointSymbol('o'), Color("#333377")])
  .lines(lx.iter(), ly.iter(), &[Color("#006633")])
  .set_title("Logistic Regression", &[]);

  fg.show();
}

