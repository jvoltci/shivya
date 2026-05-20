#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Const(f64),
    Var(usize), // Reference to the state variable by index
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Exp(Box<Expr>),
}
