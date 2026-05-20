use crate::vm::ast::Expr;

#[derive(Clone, Debug, PartialEq)]
pub enum Instruction {
    LoadConst { dest: usize, val: f64 },
    LoadVar { dest: usize, var_idx: usize },
    Add { dest: usize, src1: usize, src2: usize },
    Sub { dest: usize, src1: usize, src2: usize },
    Mul { dest: usize, src1: usize, src2: usize },
    Exp { dest: usize, src: usize },
}

pub fn compile(expr: &Expr) -> (Vec<Instruction>, usize) {
    let mut insts = Vec::new();
    let mut next_reg = 0;
    let final_reg = compile_expr(expr, &mut insts, &mut next_reg);
    (insts, final_reg)
}

fn compile_expr(expr: &Expr, insts: &mut Vec<Instruction>, next_reg: &mut usize) -> usize {
    match expr {
        Expr::Const(val) => {
            let dest = *next_reg;
            *next_reg += 1;
            insts.push(Instruction::LoadConst { dest, val: *val });
            dest
        }
        Expr::Var(var_idx) => {
            let dest = *next_reg;
            *next_reg += 1;
            insts.push(Instruction::LoadVar { dest, var_idx: *var_idx });
            dest
        }
        Expr::Add(lhs, rhs) => {
            let src1 = compile_expr(lhs, insts, next_reg);
            let src2 = compile_expr(rhs, insts, next_reg);
            let dest = *next_reg;
            *next_reg += 1;
            insts.push(Instruction::Add { dest, src1, src2 });
            dest
        }
        Expr::Sub(lhs, rhs) => {
            let src1 = compile_expr(lhs, insts, next_reg);
            let src2 = compile_expr(rhs, insts, next_reg);
            let dest = *next_reg;
            *next_reg += 1;
            insts.push(Instruction::Sub { dest, src1, src2 });
            dest
        }
        Expr::Mul(lhs, rhs) => {
            let src1 = compile_expr(lhs, insts, next_reg);
            let src2 = compile_expr(rhs, insts, next_reg);
            let dest = *next_reg;
            *next_reg += 1;
            insts.push(Instruction::Mul { dest, src1, src2 });
            dest
        }
        Expr::Exp(src_expr) => {
            let src = compile_expr(src_expr, insts, next_reg);
            let dest = *next_reg;
            *next_reg += 1;
            insts.push(Instruction::Exp { dest, src });
            dest
        }
    }
}
