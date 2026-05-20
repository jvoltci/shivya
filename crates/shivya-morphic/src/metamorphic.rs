use crate::vm::ast::Expr;
use crate::vm::compiler::compile;
use crate::vm::eval::execute;

pub struct SimpleRng {
    state: u32,
}

impl SimpleRng {
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(1664525).wrapping_add(1013904223);
        self.state
    }

    pub fn next_f64(&mut self) -> f64 {
        (self.next_u32() as f64) / (u32::MAX as f64)
    }

    pub fn choose<'a, T>(&mut self, slice: &'a [T]) -> &'a T {
        let idx = (self.next_u32() as usize) % slice.len();
        &slice[idx]
    }
}

pub fn mutate_expr(expr: &Expr, rng: &mut SimpleRng) -> Expr {
    // 15% chance to replace the sub-tree with a leaf or wrapper
    if rng.next_f64() < 0.15 {
        let choices = [0, 1, 2];
        match rng.choose(&choices) {
            0 => Expr::Const(rng.next_f64() * 5.0),
            1 => Expr::Var(if rng.next_f64() < 0.5 { 0 } else { 1 }),
            _ => Expr::Exp(Box::new(expr.clone())),
        }
    } else {
        match expr {
            Expr::Const(val) => {
                let delta = (rng.next_f64() - 0.5) * 0.5;
                Expr::Const(*val + delta)
            }
            Expr::Var(idx) => {
                Expr::Var(1 - idx) // Toggle index reference
            }
            Expr::Add(l, r) => {
                if rng.next_f64() < 0.3 {
                    Expr::Mul(Box::new(mutate_expr(l, rng)), Box::new(mutate_expr(r, rng)))
                } else {
                    Expr::Add(Box::new(mutate_expr(l, rng)), Box::new(mutate_expr(r, rng)))
                }
            }
            Expr::Sub(l, r) => {
                Expr::Sub(Box::new(mutate_expr(l, rng)), Box::new(mutate_expr(r, rng)))
            }
            Expr::Mul(l, r) => {
                if rng.next_f64() < 0.3 {
                    Expr::Add(Box::new(mutate_expr(l, rng)), Box::new(mutate_expr(r, rng)))
                } else {
                    Expr::Mul(Box::new(mutate_expr(l, rng)), Box::new(mutate_expr(r, rng)))
                }
            }
            Expr::Exp(src) => {
                Expr::Exp(Box::new(mutate_expr(src, rng)))
            }
        }
    }
}

pub struct MorphicHotSwapper {
    pub current_expr: Expr,
}

impl MorphicHotSwapper {
    pub fn new(initial: Expr) -> Self {
        Self { current_expr: initial }
    }

    pub fn run_metamorphic_step(
        &mut self,
        dataset: &[(Vec<f64>, f64)],
        seed: u32,
    ) -> bool {
        let mut rng = SimpleRng::new(seed);
        let mutated = mutate_expr(&self.current_expr, &mut rng);

        // Compile current
        let (curr_insts, curr_reg) = compile(&self.current_expr);
        // Compile mutated
        let (mut_insts, mut_reg) = compile(&mutated);

        let mut curr_error = 0.0;
        let mut mut_error = 0.0;
        let mut mut_failed = false;

        for (vars, expected) in dataset {
            let curr_res = execute(&curr_insts, vars, curr_reg);
            match curr_res {
                Ok(val) => curr_error += (val - expected).powi(2),
                Err(_) => curr_error += 1e9,
            }

            let mut_res = execute(&mut_insts, vars, mut_reg);
            match mut_res {
                Ok(val) => mut_error += (val - expected).powi(2),
                Err(_) => {
                    mut_failed = true;
                    break;
                }
            }
        }

        if !mut_failed && mut_error < curr_error {
            self.current_expr = mutated;
            true // Hot-swapped to improved expression topology!
        } else {
            false // Retained current expression
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metamorphic_mutation_and_hotswap() {
        // Initial expression: Const(1.0) * Var(0)
        let initial = Expr::Mul(
            Box::new(Expr::Const(1.0)),
            Box::new(Expr::Var(0))
        );

        let mut swapper = MorphicHotSwapper::new(initial);

        // Simple dataset target: expected = 2.0 * Var(0)
        // Dataset consists of pairs: (variables, expected_outcome)
        let dataset = vec![
            (vec![1.0], 2.0),
            (vec![2.0], 4.0),
            (vec![3.0], 6.0),
        ];

        // Run metamorphic steps. With seed variations, we expect to eventually hit
        // a mutation that changes the constant 1.0 closer to 2.0, or modifies the expression,
        // reducing the error and triggering a hot-swap!
        let mut hotswap_occurred = false;
        for seed in 0..100 {
            if swapper.run_metamorphic_step(&dataset, seed) {
                hotswap_occurred = true;
                break;
            }
        }

        // Assert that a hot-swap was triggered because we found an expression with lower error!
        assert!(hotswap_occurred, "Hot-swapper should find and swap to a better expression topology");
        assert_ne!(swapper.current_expr, Expr::Mul(Box::new(Expr::Const(1.0)), Box::new(Expr::Var(0))));
    }
}
