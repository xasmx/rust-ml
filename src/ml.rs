#[crate_type = "lib"];
#[link(name = "ml", vers = "0.1")];

extern mod la;

pub mod clustering {
  pub mod kmeans;
}

pub mod regression {
  pub mod linear;
}

pub mod opt {
  pub mod graddescent;
}
