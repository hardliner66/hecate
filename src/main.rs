use std::path::PathBuf;

use byteorder::ReadBytesExt;
use clap::{Parser, Subcommand};
use hecate_common::{CpuTrait, RunMode};
use native::{NativeCpu, NullHostIO};
use std::io::BufReader;

mod native;

#[derive(Parser)]
struct Args {
    #[arg(short, long, global = true)]
    verbose: bool,
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand)]
enum Action {
    Run { path: PathBuf },
    RunAsm { path: PathBuf },
}

fn run(memory: &[u32], verbose: bool) -> anyhow::Result<()> {
    let mut cpu = NativeCpu::new(1024 * 1024, 6, NullHostIO);
    cpu.set_verbose(verbose);

    cpu.load_memory(0, memory);

    let stats = cpu.execute(RunMode::Run)?;

    println!();
    println!(" ========== STATS ===========");
    println!();

    println!("{:#?}", stats);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let Args { action, verbose } = Args::parse();

    match action {
        Action::Run { path } => {
            let memory = std::fs::File::open(path)
                .map(|file| {
                    let mut reader = BufReader::new(file);
                    let mut values = Vec::new();
                    while let Ok(v) = reader.read_u32::<byteorder::LittleEndian>() {
                        values.push(v);
                    }
                    values
                })
                .unwrap();

            run(&memory, verbose)?;
        }

        Action::RunAsm { path } => {
            let program = std::fs::read_to_string(path)?;
            let memory = hecate_assembler::assemble_program(&program)?;

            run(&memory.data, verbose)?;
        }
    }

    Ok(())
}
