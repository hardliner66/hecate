use std::{fs::File, io::{BufReader, BufWriter}, path::PathBuf};
use byteorder::{WriteBytesExt, ReadBytesExt, LittleEndian};
use clap::Parser;
use hasm::disassembler::disassemble_program;
use hasm::assemble_program;

#[derive(Parser)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    #[clap(long, short)]
    disassemble: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Args { input, output, disassemble } = Args::parse();

    if disassemble {
        let program = read_u32_values(&input)?;
        let code = disassemble_program(&program);
        let combined_code = code.join("\n");
        std::fs::write(&output, combined_code)
            .map_err(|e| format!("Failed to write output file '{}': {e}", output.display()))?;
    } else {
        let program = std::fs::read_to_string(&input)?;
        let code = assemble_program(&program)?;
        write_u32_values(&output, &code)?;
    }

    Ok(())
}

fn read_u32_values(input: &PathBuf) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let file = File::open(input)?;
    let mut reader = BufReader::new(file);

    let mut values = Vec::new();
    while let Ok(value) = reader.read_u32::<LittleEndian>() {
        values.push(value);
    }

    Ok(values)
}

fn write_u32_values(output: &PathBuf, values: &[u32]) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(output)?;
    let mut writer = BufWriter::new(file);

    for &value in values {
        writer.write_u32::<LittleEndian>(value)?;
    }

    Ok(())
}
