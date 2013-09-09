#[crate_type = "lib"];
#[link(name = "ml", vers = "0.1")];

extern mod la;

pub mod clustering {
  pub mod kmeans;
}
