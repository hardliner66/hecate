use clap::Parser;
use hecate_assembler::{Assembler, Disassembler};
use hecate_common::BytecodeFile;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    #[clap(long, short)]
    disassemble: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Args {
        input,
        output,
        disassemble,
    } = Args::parse();

    if disassemble {
        let program = BytecodeFile::load(input)?;
        let disassembler = Disassembler::from_bytecode_file(&program);
        let code = disassembler.disassemble_program(&program.data)?;
        std::fs::write(&output, code)
            .map_err(|e| format!("Failed to write output file '{}': {e}", output.display()))?;
    } else {
        let program = std::fs::read_to_string(&input)?;
        let mut assembler = Assembler::new();
        let code = assembler.assemble_program(&program)?;
        code.save(output)?;
    }

    Ok(())
}
