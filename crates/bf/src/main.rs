use std::path::PathBuf;

use clap::Parser;
use fck::{lexer::lex, parser::parse};

#[derive(Parser)]
struct Args {
    input: PathBuf,
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let tokens = lex("source code")?;
    let ast = parse(&tokens)?;
    println!("{:?}", ast);

    Ok(())
}
