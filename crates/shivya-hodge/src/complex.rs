use crate::operators::SparseMatrix;

#[derive(Clone, Debug)]
pub struct SimplicialStateComplex {
    pub vertices: Vec<String>,
    pub edges: Vec<(usize, usize)>,
    pub triangles: Vec<(usize, usize, usize)>,
    pub vertex_states: Vec<f64>,
    pub edge_states: Vec<f64>,
}

impl SimplicialStateComplex {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            edges: Vec::new(),
            triangles: Vec::new(),
            vertex_states: Vec::new(),
            edge_states: Vec::new(),
        }
    }

    pub fn get_vertex_index(&self, label: &str) -> Option<usize> {
        self.vertices.iter().position(|v| v == label)
    }

    pub fn add_vertex(&mut self, label: &str, initial_state: f64) -> usize {
        if let Some(idx) = self.get_vertex_index(label) {
            idx
        } else {
            self.vertices.push(label.to_string());
            self.vertex_states.push(initial_state);
            self.vertices.len() - 1
        }
    }

    pub fn add_edge(&mut self, u_label: &str, v_label: &str, initial_state: f64) {
        let u = self.add_vertex(u_label, 0.0);
        let v = self.add_vertex(v_label, 0.0);
        
        // Avoid duplicate edges
        if self.edges.iter().any(|&(a, b)| (a == u && b == v) || (a == v && b == u)) {
            return;
        }
        
        self.edges.push((u, v));
        self.edge_states.push(initial_state);
        self.rebuild_triangles();
    }

    fn rebuild_triangles(&mut self) {
        self.triangles.clear();
        let n = self.vertices.len();
        // Look for triples {u, v, w} where u < v < w
        for u in 0..n {
            for v in (u + 1)..n {
                for w in (v + 1)..n {
                    let uv = self.has_any_edge(u, v);
                    let vw = self.has_any_edge(v, w);
                    let uw = self.has_any_edge(u, w);
                    if uv && vw && uw {
                        self.triangles.push((u, v, w));
                    }
                }
            }
        }
    }

    fn has_any_edge(&self, u: usize, v: usize) -> bool {
        self.edges.iter().any(|&(a, b)| (a == u && b == v) || (a == v && b == u))
    }

    pub fn find_edge_index(&self, u: usize, v: usize) -> Option<(usize, f64)> {
        for (i, &(a, b)) in self.edges.iter().enumerate() {
            if a == u && b == v {
                return Some((i, 1.0));
            } else if a == v && b == u {
                return Some((i, -1.0));
            }
        }
        None
    }

    pub fn d0(&self) -> SparseMatrix {
        let mut d = SparseMatrix::new(self.edges.len(), self.vertices.len());
        for (i, &(u, v)) in self.edges.iter().enumerate() {
            d.insert(i, u, -1.0);
            d.insert(i, v, 1.0);
        }
        d
    }

    pub fn d1(&self) -> SparseMatrix {
        let mut d = SparseMatrix::new(self.triangles.len(), self.edges.len());
        for (j, &(u, v, w)) in self.triangles.iter().enumerate() {
            // boundary of [u, v, w] is [v, w] - [u, w] + [u, v]
            if let Some((e_vw, sign_vw)) = self.find_edge_index(v, w) {
                d.insert(j, e_vw, sign_vw);
            }
            if let Some((e_uw, sign_uw)) = self.find_edge_index(u, w) {
                d.insert(j, e_uw, -sign_uw);
            }
            if let Some((e_uv, sign_uv)) = self.find_edge_index(u, v) {
                d.insert(j, e_uv, sign_uv);
            }
        }
        d
    }
}
