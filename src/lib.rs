#![crate_type = "lib"]
#![crate_id = "ml#0.1"]

#![feature(globs)]
#![feature(phase)]

extern crate gnuplot;

#[phase(plugin, link)] extern crate la;

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
