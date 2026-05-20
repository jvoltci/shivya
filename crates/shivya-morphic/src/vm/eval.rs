use crate::vm::compiler::Instruction;

pub fn execute(
    instructions: &[Instruction],
    variables: &[f64],
    final_reg: usize,
) -> Result<f64, String> {
    let mut registers = vec![0.0; final_reg + 1];
    let mut budget = 0;

    for inst in instructions {
        budget += 1;
        if budget > 500 {
            return Err("Execution aborted: instruction budget exceeded limit of 500 cycles".to_string());
        }

        match *inst {
            Instruction::LoadConst { dest, val } => {
                if dest >= registers.len() {
                    return Err(format!("Register index out of bounds: {}", dest));
                }
                registers[dest] = val;
            }
            Instruction::LoadVar { dest, var_idx } => {
                if dest >= registers.len() {
                    return Err(format!("Register index out of bounds: {}", dest));
                }
                if var_idx >= variables.len() {
                    return Err(format!("Variable index out of bounds: {}", var_idx));
                }
                registers[dest] = variables[var_idx];
            }
            Instruction::Add { dest, src1, src2 } => {
                if dest >= registers.len() || src1 >= registers.len() || src2 >= registers.len() {
                    return Err("Register index out of bounds during Add".to_string());
                }
                registers[dest] = registers[src1] + registers[src2];
            }
            Instruction::Sub { dest, src1, src2 } => {
                if dest >= registers.len() || src1 >= registers.len() || src2 >= registers.len() {
                    return Err("Register index out of bounds during Sub".to_string());
                }
                registers[dest] = registers[src1] - registers[src2];
            }
            Instruction::Mul { dest, src1, src2 } => {
                if dest >= registers.len() || src1 >= registers.len() || src2 >= registers.len() {
                    return Err("Register index out of bounds during Mul".to_string());
                }
                registers[dest] = registers[src1] * registers[src2];
            }
            Instruction::Exp { dest, src } => {
                if dest >= registers.len() || src >= registers.len() {
                    return Err("Register index out of bounds during Exp".to_string());
                }
                registers[dest] = registers[src].exp();
            }
        }
    }

    if final_reg >= registers.len() {
        return Err("Final register out of bounds".to_string());
    }

    Ok(registers[final_reg])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::ast::Expr;
    use crate::vm::compiler::compile;

    #[test]
    fn test_ast_eval_basic() {
        // (2.5 + Var(0)) * Const(2.0)
        let expr = Expr::Mul(
            Box::new(Expr::Add(
                Box::new(Expr::Const(2.5)),
                Box::new(Expr::Var(0))
            )),
            Box::new(Expr::Const(2.0))
        );

        let (insts, final_reg) = compile(&expr);
        let vars = vec![1.5]; // Var(0) = 1.5 -> expected output: (2.5 + 1.5) * 2 = 8.0
        let res = execute(&insts, &vars, final_reg).unwrap();
        assert!((res - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_instruction_budget_exceeded() {
        // Build a highly nested AST structure that exceeds 500 instructions
        let mut expr = Expr::Const(1.0);
        for _ in 0..600 {
            expr = Expr::Add(Box::new(expr), Box::new(Expr::Const(0.1)));
        }

        let (insts, final_reg) = compile(&expr);
        let res = execute(&insts, &[], final_reg);
        assert!(res.is_err());
        assert!(res.unwrap_err().contains("budget exceeded"));
    }
}
