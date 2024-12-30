use std::path::PathBuf;

use clap::{Parser, Subcommand};
use hecate_common::{BytecodeFile, CpuTrait, RunMode};
use native::{HostIO, NativeCpu};

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

#[derive(Debug)]
struct SimpleHostIo;
impl HostIO for SimpleHostIo {
    fn syscall(
        &mut self,
        code: u32,
        cpu: &mut NativeCpu<Self>,
    ) -> Result<usize, hecate_common::ExecutionError>
    where
        Self: Sized,
    {
        match code {
            0 => {
                let start = cpu.get_registers()[1] as usize;
                let length = cpu.get_registers()[2] as usize;
                let mem = &cpu.get_memory()[start..start + length];
                let s = String::from_utf8(
                    mem.into_iter()
                        .map(|v| u8::from_le_bytes([v.to_le_bytes()[0]]))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                println!("{s}");
                Ok(2500 + (length * 300))
            }
            _ => Err(hecate_common::ExecutionError::InvalidSyscall(code)),
        }
    }
}

fn run(memory: &[u32], entrypoint: u32, verbose: bool) -> anyhow::Result<()> {
    let mut cpu = NativeCpu::new(1024 * 1024, 16, SimpleHostIo);
    cpu.set_verbose(verbose);
    cpu.set_entrypoint(entrypoint);

    cpu.load_protected_memory(0, memory);

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
            let file = BytecodeFile::load(path).unwrap();

            run(&file.data, file.header.entrypoint, verbose)?;
        }

        Action::RunAsm { path } => {
            let program = std::fs::read_to_string(path)?;
            let mut assembler = hecate_assembler::Assembler::new();
            let memory = assembler.assemble_program(&program)?;

            run(&memory.data, memory.header.entrypoint, verbose)?;
        }
    }

    Ok(())
}
