#[derive(Clone, Debug)]
pub struct MarkovBlanket<const S_DIM: usize, const A_DIM: usize, const I_DIM: usize> {
    pub sensory: [f64; S_DIM],
    pub active: [f64; A_DIM],
    pub internal: [f64; I_DIM],
}

impl<const S_DIM: usize, const A_DIM: usize, const I_DIM: usize> MarkovBlanket<S_DIM, A_DIM, I_DIM> {
    pub fn new() -> Self {
        Self {
            sensory: [0.0; S_DIM],
            active: [0.0; A_DIM],
            internal: [0.0; I_DIM],
        }
    }

    pub fn update_sensory(&mut self, s: &[f64; S_DIM]) {
        self.sensory.copy_from_slice(s);
    }

    pub fn update_active(&mut self, a: &[f64; A_DIM]) {
        self.active.copy_from_slice(a);
    }

    pub fn update_internal(&mut self, mu: &[f64; I_DIM]) {
        self.internal.copy_from_slice(mu);
    }
}
