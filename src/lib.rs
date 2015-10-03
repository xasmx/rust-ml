extern crate gnuplot;
#[macro_use]
extern crate la;
extern crate num;

pub use classification::logistic::LogisticRegression;
pub use clustering::kmeans::KMeans;
pub use regression::linear::LinearRegression;

pub mod graph;
pub mod opt;

mod classification {
  pub mod logistic;
}

mod clustering {
  pub mod kmeans;
}

mod regression {
  pub mod linear;
}
