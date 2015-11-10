extern crate gnuplot;
extern crate la;
extern crate ml;

use std::str::FromStr;

use gnuplot::*;
use la::Matrix;
use la::util::read_csv;
use ml::LinearRegression;
use ml::graph;

fn main() {
  let data = read_csv("example_data/linreg.csv", &|s| {
    FromStr::from_str(s).unwrap()
  });

  let x = data.get_columns(0..(data.cols() - 1));
  let y = data.get_columns(data.cols() - 1);
  let mut cost_history = vec![];

  let lr = LinearRegression::new()
    .learning_rate(0.005f64)
    .max_iterations(100)
    .cost_history(&mut |cost| { cost_history.push(cost); })
    .train(&x, &y);

  let mut lx = vec![0.0f64; 2];
  let mut ly = vec![0.0f64; 2];
  for i in 0..lx.len() {
    lx[i] = (i as f64) * 30.0f64;
    ly[i] = lr.predict(&Matrix::new(1, 1, vec![lx[i]]));
  }

  let mut fg = Figure::new();
  graph::draw_cost_graph(&mut fg, &cost_history);
  fg.show();

  let mut fg = Figure::new();

  fg.axes2d()
    .set_aspect_ratio(Fix(1.0))
    .lines(lx.iter(), ly.iter(), &[Color("#006633")])
    .points(x.get_data().iter(), y.get_data().iter(), &[PointSymbol('x'), Color("#ffaa77")])
    .set_title("Linear Regression", &[]);

  fg.show();
}

