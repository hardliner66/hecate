use std::{io::BufWriter, path::PathBuf};

use byteorder::WriteBytesExt;
use clap::Parser;
use hasm::assemble_program;

#[derive(Parser)]
struct Args {
    input: PathBuf,
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Args { input, output } = Args::parse();
    let program = std::fs::read_to_string(input)?;

    let code = assemble_program(&program)?;

    let file = std::fs::File::create(output)?;
    let mut reader = BufWriter::new(file);
    for v in code {
        reader.write_u32::<byteorder::LittleEndian>(v)?;
    }
    Ok(())
}
