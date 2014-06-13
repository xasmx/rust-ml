use std::io::process::{Command};
use std::io;

use classification::logistic::LogisticRegression;
use la::matrix::*;

pub fn show_decision_boundary_2d(lr : &LogisticRegression) {
  let mut out : Vec<u8> = Vec::new();

  write_str(&mut out, "set termoption enhanced\n");
  write_str(&mut out, "set size ratio 1e0\n");
  write_str(&mut out, "set title \"Logistic Regression\"\n");
  write_str(&mut out, "set contour base\n");
  write_str(&mut out, "set cntrparam bspline\n");
  write_str(&mut out, "set cntrparam levels discrete 0\n");
  write_str(&mut out, "set view map\n");
  write_str(&mut out, "unset surface\n");
  write_str(&mut out, "splot \"-\" with lines\n");


  for row in range(30, 100) {
    for col in range(30, 100) {
      write_f64(&mut out, row as f64);
      write_str(&mut out, " ");
      write_f64(&mut out, col as f64);
      write_str(&mut out, " ");

      let x = vector(vec![col as f64, row as f64]);
      let p = lr.p(&x) - lr.get_threshold();
      write_f64(&mut out, p);
      write_str(&mut out, "\n");
    }
    write_str(&mut out, "\n");
  }
  write_str(&mut out, "e\n");

  let mut p = match Command::new("gnuplot").arg("-p").spawn() {
    Ok(p) => p,
    Err(e) => fail!("failed to execute process: {}", e),
  };

  let mut input = p.stdin.take_unwrap();
  let _ = input.write(out.as_slice());
  io::println(String::from_utf8(out).unwrap().as_slice());
}

fn write_str(out : &mut Vec<u8>, s: &str) {
  out.push_all(s.as_bytes());
}

//fn write_int(out : &mut Vec<u8>, i: int) {
//  write_str(out, i.to_str().as_slice());
//}

fn write_f64(out : &mut Vec<u8>, f: f64) {
  write_str(out, f.to_str().as_slice());
}
