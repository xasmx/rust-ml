#[crate_type = "lib"];
#[link(name = "ml", vers = "0.1")];

extern mod gnuplot;
extern mod la;

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

