use std::{io::BufWriter, path::PathBuf};

use byteorder::{ReadBytesExt, WriteBytesExt};
use clap::{Parser, Subcommand};
use common::{CpuTrait, RunMode};
use native::{Bytecode, NativeCpu};
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
    RunDemo,
    WriteDemo { path: PathBuf },
}

const DEMO: &[u32] = {
    use Bytecode::*;
    &[
        /* 0 */ PushValue as u32,
        /* 1 */ 42,
        /* 2 */ LoadValue as u32,
        /* 3 */ 0,
        /* 4 */ 10,
        /* 5 */ PushReg as u32,
        /* 6 */ 0,
        /* 7 */ Call as u32,
        /* 8 */ 17,
        /* 9 */ Pop as u32,
        /* 10 */ 3,
        /* 11 */ Store as u32,
        /* 12 */ 40,
        /* 13 */ 3,
        /* 14 */ Inspect as u32,
        /* 15 */ 40,
        /* 16 */ Halt as u32,
        /* 17 */ Pop as u32,
        /* 18 */ 5,
        /* 19 */ Pop as u32,
        /* 20 */ 0,
        /* 21 */ Pop as u32,
        /* 22 */ 1,
        /* 23 */ Add as u32,
        /* 24 */ 0,
        /* 25 */ 1,
        /* 26 */ PushReg as u32,
        /* 27 */ 5,
        /* 28 */ RetReg as u32,
        /* 29 */ 0,
    ]
};

fn main() -> anyhow::Result<()> {
    let Args { action, verbose } = Args::parse();

    match action {
        Action::RunDemo => {
            let mut cpu = NativeCpu::new(1024 * 1024, 6);
            cpu.set_verbose(verbose);

            cpu.load_memory(0, DEMO);

            let stats = cpu.execute(RunMode::Run)?;

            println!();
            println!(" ========== STATS ===========");
            println!();

            if verbose {
                println!(
                    "Total cycles: {}, Memory Access score: {}",
                    stats.cycles, stats.memory_access_score
                );
            }

            println!("Total Score: {}", stats.cycles + stats.memory_access_score);
        }
        Action::WriteDemo { path } => {
            let file = std::fs::File::create(path)?;
            let mut writer = BufWriter::new(file);
            for &v in DEMO {
                writer.write_u32::<byteorder::LittleEndian>(v)?;
            }
        }

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

            let mut cpu = NativeCpu::new(1024 * 1024, 6);
            cpu.set_verbose(verbose);

            cpu.load_memory(0, &memory);

            let stats = cpu.execute(RunMode::Run)?;

            println!();
            println!(" ========== STATS ===========");
            println!();

            if verbose {
                println!(
                    "Total cycles: {}, Memory Access score: {}",
                    stats.cycles, stats.memory_access_score
                );
            }

            println!("Total Score: {}", stats.cycles + stats.memory_access_score);
        }
    }

    Ok(())
}
