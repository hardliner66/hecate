use std::path::PathBuf;

use clap::Parser;
use hecate_bf::compile_program;

#[derive(Parser)]
struct Args {
    input: PathBuf,
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let Args { input, output } = Args::parse();

    let code = std::fs::read_to_string(input)?;
    let file = compile_program(&code)?;
    file.save(output).unwrap();

    Ok(())
}
