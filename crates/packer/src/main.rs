use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use clap::Parser;
use hecate_assembler::Disassembler;
use hecate_common::BytecodeFile;
use std::{
    fs::File,
    io::{BufReader, BufWriter},
    path::PathBuf,
};

#[derive(Parser)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    #[clap(long, short, default_value = "0")]
    entrypoint: u32,
    #[clap(long, short)]
    /// labels and their address in the format: <label_name>:<address>
    labels: Vec<String>,
    #[clap(long, short)]
    unpack: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let Args {
        input,
        output,
        entrypoint,
        labels,
        unpack,
    } = Args::parse();

    if unpack {
        let program = BytecodeFile::load(input)?;
        println!("Entrypoint: {}", program.header.entrypoint);
        println!("Labels:");
        for (label, address) in program.header.labels {
            println!("  - {}:{}", label, address);
        }
        write_u32_values(&output, &program.data)
            .map_err(|e| format!("Failed to write output file '{}': {e}", output.display()))?;
    } else {
        let code = read_u32_values(&input)?;
        let disassembler = Disassembler::new();
        _ = disassembler.disassemble_program(&code)?;
        let mut code = BytecodeFile::new(code);
        code.header.entrypoint = entrypoint;
        code.header.labels = labels
            .into_iter()
            .map(|s| {
                let (a, b) = s.split_once(":").unwrap();
                (a.to_owned(), b.to_owned())
            })
            .map(|(name, address)| (name, address.parse().unwrap()))
            .collect();
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
