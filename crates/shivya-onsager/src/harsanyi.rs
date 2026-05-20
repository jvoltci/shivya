pub struct LocalCoalitionSolver {
    pub center_id: usize,
    pub neighbors: Vec<usize>,
}

impl LocalCoalitionSolver {
    pub fn new(center_id: usize, neighbors: Vec<usize>) -> Self {
        Self { center_id, neighbors }
    }

    pub fn get_all_masks(&self) -> Vec<u8> {
        let num_nodes = 1 + self.neighbors.len();
        let total_combinations = 1 << num_nodes;
        let mut masks = Vec::with_capacity(total_combinations - 1);
        for m in 1..total_combinations {
            masks.push(m as u8);
        }
        masks
    }

    pub fn mask_to_nodes(&self, mask: u8) -> Vec<usize> {
        let mut nodes = Vec::new();
        if (mask & 1) != 0 {
            nodes.push(self.center_id);
        }
        for (idx, &nbr) in self.neighbors.iter().enumerate() {
            if (mask & (1 << (idx + 1))) != 0 {
                nodes.push(nbr);
            }
        }
        nodes
    }

    pub fn calculate_dividends<F>(&self, val_fn: F) -> Vec<(u8, f64)>
    where
        F: Fn(u8) -> f64,
    {
        let masks = self.get_all_masks();
        let mut sorted_masks = masks.clone();
        sorted_masks.sort_by_key(|&m| m.count_ones());

        let limit = 1 << (1 + self.neighbors.len());
        let mut dividends = vec![0.0; limit];

        for &mask in &sorted_masks {
            let val = val_fn(mask);
            let mut sum_sub_dividends = 0.0;
            // Iterate over all strict non-empty subsets of mask using bit operations
            let mut sub = (mask - 1) & mask;
            while sub > 0 {
                sum_sub_dividends += dividends[sub as usize];
                sub = (sub - 1) & mask;
            }
            dividends[mask as usize] = val - sum_sub_dividends;
        }

        let mut results = Vec::new();
        for &mask in &masks {
            results.push((mask, dividends[mask as usize]));
        }
        results
    }
}
