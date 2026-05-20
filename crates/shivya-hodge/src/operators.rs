#[derive(Clone, Debug)]
pub struct SparseMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Vec<(usize, f64)>>, // row index -> list of (col_index, value)
}

impl SparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![Vec::new(); rows],
        }
    }

    pub fn insert(&mut self, r: usize, c: usize, val: f64) {
        if r >= self.rows || c >= self.cols {
            panic!("Index out of bounds: ({}, {}) for dimensions {}x{}", r, c, self.rows, self.cols);
        }
        // If entry already exists, update it, otherwise push
        if let Some(entry) = self.data[r].iter_mut().find(|(col, _)| *col == c) {
            entry.1 = val;
        } else {
            self.data[r].push((c, val));
        }
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        if r >= self.rows || c >= self.cols {
            return 0.0;
        }
        self.data[r]
            .iter()
            .find(|(col, _)| *col == c)
            .map(|(_, val)| *val)
            .unwrap_or(0.0)
    }

    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.cols, "Vector length must match matrix columns");
        let mut y = vec![0.0; self.rows];
        for r in 0..self.rows {
            let mut sum = 0.0;
            for &(c, val) in &self.data[r] {
                sum += val * x[c];
            }
            y[r] = sum;
        }
        y
    }

    pub fn transpose(&self) -> Self {
        let mut t = Self::new(self.cols, self.rows);
        for r in 0..self.rows {
            for &(c, val) in &self.data[r] {
                t.insert(c, r, val);
            }
        }
        t
    }

    // Multiply two sparse matrices: self * other
    pub fn mul_mat(&self, other: &Self) -> Self {
        assert_eq!(self.cols, other.rows, "Matrix dimensions mismatch for multiplication");
        let mut result = Self::new(self.rows, other.cols);
        for r in 0..self.rows {
            for &(c_self, val_self) in &self.data[r] {
                for &(c_other, val_other) in &other.data[c_self] {
                    let current = result.get(r, c_other);
                    result.insert(r, c_other, current + val_self * val_other);
                }
            }
        }
        result
    }
}
