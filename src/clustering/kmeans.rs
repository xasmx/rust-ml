use std::num::Float;
use std::vec::Vec;

use la::Matrix;

pub struct KMeans {
  assignments : Vec<uint>
}

impl KMeans {
  /// Performs K-means clustering on the passed dataset.
  /// Returns a vector of assignments, assigning each data row
  /// to the specific cluster [0, k).
  ///
  /// ```ignore
  /// #![feature(phase)]
  /// #[phase(plugin, link)] extern crate la;
  /// ...
  /// let m = m!(1.0, 2.0; 3.0, 4.0; 5.0, 6.0; 7.0, 8.0);
  /// let assignments = KMeans::cluster(1, &m);
  /// ```
  pub fn cluster(k : uint, m : &Matrix<f64>) -> KMeans {
    assert!(k >= 1);

    let mut means = init(k, m);
    let mut assignments = Vec::from_elem(m.rows(), 0u);
    perform_assignments(&mut means, &mut assignments, m, false);
    loop {
      update_means(&mut means, &mut assignments, m);
      if !perform_assignments(&mut means, &mut assignments, m, true) {
        break;
      }
    }

    KMeans {
      assignments : assignments
    }
  }

  pub fn get_assignments<'a>(&'a self) -> &'a Vec<uint> {
    &self.assignments
  }
}

fn perform_assignments(means : &mut Matrix<f64>, assignments : &mut Vec<uint>, m : &Matrix<f64>, check_assignments_flag : bool) -> bool {
  let mut assignments_changed = false;

  for data_row in range(0, m.rows()) {
    let mut min_d = norm(means, 0, m, data_row);
    let mut min_mean_idx = 0;
    for mean_row in range(1, means.rows()) {
      let d = norm(means, mean_row, m, data_row);
      if d < min_d {
        min_d = d;
        min_mean_idx = mean_row;
      }
    }

    if check_assignments_flag {
      if min_mean_idx != *assignments.get(data_row) {
        assignments_changed = true;
        *assignments.get_mut(data_row) = min_mean_idx;
      }
    } else {
      *assignments.get_mut(data_row) = min_mean_idx;
    }
  }

  assignments_changed
}

fn update_means(means : &mut Matrix<f64>, assignments : &mut Vec<uint>, m : &Matrix<f64>) {
  let mut data_count = Vec::from_elem(means.cols(), 0);
  for i in range(0, means.get_data().len()) {
    *means.get_mut_data().get_mut(i) = 0.0;
  }

  let mut row_idx = 0;
  for row in range(0, m.rows()) {
    let assignment = *assignments.get(row);

    let mean_row_idx = assignment * means.cols();
    for col in range(0, m.cols()) {
      *means.get_mut_data().get_mut(mean_row_idx + col) += *m.get_data().get(row_idx + col);
    }

    *data_count.get_mut(assignment) += 1;
    row_idx += m.cols();
  }

  let mut row_idx = 0;
  for row in range(0, means.rows()) {
    for col in range(0, means.cols()) {
      *means.get_mut_data().get_mut(row_idx + col) /= *data_count.get(row) as f64;
    }

    row_idx += means.cols();
  }
}

fn norm(means : &mut Matrix<f64>, mean_row : uint, m : &Matrix<f64>, data_row : uint) -> f64 {
  let mut sum = 0.0;
  for col in range(0, means.cols()) {
    let diff = m.get(data_row, col) - means.get(mean_row, col);
    sum += diff * diff;
  }
  sum
}

fn bounds(m : &Matrix<f64>) -> (Vec<f64>, Vec<f64>) {
  let mut min_data : Vec<f64> = Vec::from_elem(m.cols(), Float::infinity());
  let mut max_data : Vec<f64> = Vec::from_elem(m.cols(), Float::neg_infinity());
  let mut col_idx = 0;
  for i in range(0, m.get_data().len()) {
    let v = *m.get_data().get(i);
    if v < *min_data.get(col_idx) {
      *min_data.get_mut(col_idx) = v;
    }
    if v > *max_data.get(col_idx) {
      *max_data.get_mut(col_idx) = v;
    }
    col_idx += 1;
    col_idx %= m.cols();
  }

  (min_data, max_data)
}

fn init(k : uint, m : &Matrix<f64>) -> Matrix<f64> {
  let (min_data, max_data) = bounds(m);
  let mut means : Matrix<f64> = Matrix::random(k, m.cols());
  for row in range(0, means.rows()) {
    for col in range(0, means.cols()) {
      let deviation = *max_data.get(col) - *min_data.get(col);
      let v = means.get(row, col);
      means.set(row, col, *min_data.get(col) + deviation * v);
    }
  }
  means
}

#[test]
fn test_kmeans() {
  let m = m!(1.0, 2.0; 3.0, 4.0);
  let _km = KMeans::cluster(1, &m);
}

