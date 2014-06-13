#![crate_type = "lib"]
#![crate_id = "ml#0.1"]

#![feature(globs)]

extern crate gnuplot;
extern crate la;

pub mod classification {
  pub mod logistic;
}

pub mod clustering {
  pub mod kmeans;
}

pub mod regression {
  pub mod linear;
}

pub mod opt {
  pub mod graddescent;
}

pub mod graph {
  pub mod costgraph;
  pub mod decisionboundary;
}

