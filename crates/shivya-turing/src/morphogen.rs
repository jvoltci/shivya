pub struct MorphogenSystem {
    pub max_nodes: usize,
    pub active: Vec<bool>,
    
    // Activator and inhibitor concentrations
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    
    // Adjacency matrix for topology weights
    pub adj: Vec<Vec<f64>>,
    
    // Constants
    pub d_u: f64, // Activator diffusion rate
    pub d_v: f64, // Inhibitor diffusion rate
    
    // Gierer-Meinhardt parameter constants
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

impl MorphogenSystem {
    pub fn new(max_nodes: usize, d_u: f64, d_v: f64) -> Self {
        Self {
            max_nodes,
            active: vec![false; max_nodes],
            u: vec![0.0; max_nodes],
            v: vec![0.0; max_nodes],
            adj: vec![vec![0.0; max_nodes]; max_nodes],
            d_u,
            d_v,
            a: 0.1,
            b: 1.0,
            c: 1.0,
            d: 1.0,
        }
    }

    pub fn activate_node(&mut self, idx: usize, u_init: f64, v_init: f64) {
        if idx < self.max_nodes {
            self.active[idx] = true;
            self.u[idx] = u_init;
            self.v[idx] = v_init;
        }
    }

    pub fn set_edge(&mut self, u_idx: usize, v_idx: usize, weight: f64) {
        if u_idx < self.max_nodes && v_idx < self.max_nodes {
            self.adj[u_idx][v_idx] = weight;
            self.adj[v_idx][u_idx] = weight; // Symmetric undirected connections
        }
    }

    // Calculates maximum node degree to dynamically clamp dt via CFL guard
    pub fn get_cfl_dt(&self, requested_dt: f64) -> f64 {
        let mut max_deg = 0.0;
        for i in 0..self.max_nodes {
            if self.active[i] {
                let mut deg = 0.0;
                for j in 0..self.max_nodes {
                    if self.active[j] {
                        deg += self.adj[i][j];
                    }
                }
                if deg > max_deg {
                    max_deg = deg;
                }
            }
        }
        
        let max_diffusion = self.d_u.max(self.d_v);
        // CFL requirement: dt < 0.5 / (max_diffusion * max_degree)
        let safety_limit = 0.45 / (max_diffusion * max_deg.max(1.0));
        requested_dt.min(safety_limit)
    }

    // Computes Derivatives du/dt and dv/dt
    pub fn compute_derivatives(&self, u_state: &[f64], v_state: &[f64], du: &mut [f64], dv: &mut [f64]) {
        for i in 0..self.max_nodes {
            if !self.active[i] {
                du[i] = 0.0;
                dv[i] = 0.0;
                continue;
            }

            // 1. Calculate discrete Laplacian for node i
            let mut lap_u = 0.0;
            let mut lap_v = 0.0;
            for j in 0..self.max_nodes {
                if self.active[j] && self.adj[i][j] > 0.0 {
                    let weight = self.adj[i][j];
                    lap_u += weight * (u_state[j] - u_state[i]);
                    lap_v += weight * (v_state[j] - v_state[i]);
                }
            }

            // 2. Reaction kinetics (Gierer-Meinhardt) with division safeguard
            let ui = u_state[i];
            let vi = v_state[i].max(1e-5); // Prevent div by zero

            let react_u = self.a - self.b * ui + (ui * ui) / vi;
            let react_v = self.c * ui * ui - self.d * vi;

            du[i] = self.d_u * lap_u + react_u;
            dv[i] = self.d_v * lap_v + react_v;
        }
    }

    // Runge-Kutta 4th Order step
    pub fn step_rk4(&mut self, requested_dt: f64) {
        let dt = self.get_cfl_dt(requested_dt);
        let n = self.max_nodes;

        let mut k1_u = vec![0.0; n];
        let mut k1_v = vec![0.0; n];
        self.compute_derivatives(&self.u, &self.v, &mut k1_u, &mut k1_v);

        let mut tmp_u = vec![0.0; n];
        let mut tmp_v = vec![0.0; n];
        for i in 0..n {
            tmp_u[i] = self.u[i] + 0.5 * dt * k1_u[i];
            tmp_v[i] = self.v[i] + 0.5 * dt * k1_v[i];
        }

        let mut k2_u = vec![0.0; n];
        let mut k2_v = vec![0.0; n];
        self.compute_derivatives(&tmp_u, &tmp_v, &mut k2_u, &mut k2_v);

        for i in 0..n {
            tmp_u[i] = self.u[i] + 0.5 * dt * k2_u[i];
            tmp_v[i] = self.v[i] + 0.5 * dt * k2_v[i];
        }

        let mut k3_u = vec![0.0; n];
        let mut k3_v = vec![0.0; n];
        self.compute_derivatives(&tmp_u, &tmp_v, &mut k3_u, &mut k3_v);

        for i in 0..n {
            tmp_u[i] = self.u[i] + dt * k3_u[i];
            tmp_v[i] = self.v[i] + dt * k3_v[i];
        }

        let mut k4_u = vec![0.0; n];
        let mut k4_v = vec![0.0; n];
        self.compute_derivatives(&tmp_u, &tmp_v, &mut k4_u, &mut k4_v);

        // Update concentrations using weighted average of stages
        for i in 0..n {
            if self.active[i] {
                self.u[i] += (dt / 6.0) * (k1_u[i] + 2.0 * k2_u[i] + 2.0 * k3_u[i] + k4_u[i]);
                self.v[i] += (dt / 6.0) * (k1_v[i] + 2.0 * k2_v[i] + 2.0 * k3_v[i] + k4_v[i]);
            }
        }
    }
}
