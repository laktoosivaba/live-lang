use chumsky::Parser;
use crate::frontend::hydra_ecma::*;
use crate::frontend::hydra_js::*;

mod frontend;
mod backend;

fn main() {
    // Parse some Brainfuck with our parser
    let ast = hydra_js().parse("--[>--->->->++>-<<<<<-------]>--.>---------.>--..+++.>----.>+++++++++.<<.+++.------.<-.>>+.");

    // println!("{:?}", ast.unwrap());

    hydra_ecma();
}
