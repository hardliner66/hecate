use std::path::PathBuf;

use hecate_common::{
    native::{HostIO, NativeCpu},
    BytecodeFile, RunMode,
};

use clap::{Parser, Subcommand};
use macroquad::prelude::*;

#[derive(Debug)]
struct TurtleState {
    x: f64,           // Current X position
    y: f64,           // Current Y position
    heading: f64,     // Current direction (angle in degrees)
    pen_down: bool,   // Whether the pen is drawing
    pen_color: Color, // Current pen color
    pen_size: f32,    // Pen thickness
    speed: u32,
}

impl TurtleState {
    pub fn new() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            heading: 0.0,
            pen_down: true,
            pen_color: WHITE,
            pen_size: 2.0,
            speed: 60,
        }
    }

    pub fn move_forward(&mut self, distance: f64) {
        let rad = self.heading.to_radians();
        let new_x = self.x + distance * rad.cos();
        let new_y = self.y - distance * rad.sin(); // Y-axis is inverted in most 2D graphics

        if self.pen_down {
            draw_line(
                self.x as f32,
                self.y as f32,
                new_x as f32,
                new_y as f32,
                self.pen_size,
                self.pen_color,
            );
        }

        self.x = new_x;
        self.y = new_y;
    }

    pub fn move_backward(&mut self, distance: f64) {
        self.move_forward(-distance);
    }

    pub fn turn_right(&mut self, angle: f64) {
        self.heading = (self.heading - angle) % 360.0;
    }

    pub fn turn_left(&mut self, angle: f64) {
        self.heading = (self.heading + angle) % 360.0;
    }

    pub fn pen_up(&mut self) {
        self.pen_down = false;
    }

    pub fn pen_down(&mut self) {
        self.pen_down = true;
    }

    pub fn set_pen_color(&mut self, color: Color) {
        self.pen_color = color;
    }

    pub fn set_speed(&mut self, speed: u32) {
        self.speed = speed;
    }

    pub fn set_pen_size(&mut self, size: f32) {
        self.pen_size = size;
    }

    pub fn clear_screen(&mut self) {
        clear_background(BLACK);
    }

    pub fn reset(&mut self) {
        self.x = screen_width() as f64 / 2.0;
        self.y = screen_height() as f64 / 2.0;
        self.heading = 0.0;
        self.pen_down = true;
        self.pen_color = WHITE;
        self.pen_size = 2.0;
        self.clear_screen();
    }
}

#[derive(Debug)]
struct TurtleIo {
    turtle: TurtleState, // This struct will hold the state of the turtle, like position, heading, pen state, etc.
}

impl TurtleIo {
    pub fn new() -> Self {
        Self {
            turtle: TurtleState::new(),
        }
    }
}

impl HostIO for TurtleIo {
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
                // Print string (already implemented)
                let start = cpu.get_registers()[2] as usize;
                let length = cpu.get_registers()[3] as usize;
                let mem = &cpu.get_memory()[start..start + length];
                let s = String::from_utf8(
                    mem.iter()
                        .map(|v| u8::from_le_bytes([v.to_le_bytes()[0]]))
                        .collect::<Vec<_>>(),
                )?;
                eprint!("{s}");
                Ok(2500 + (length * 300))
            }
            1 => {
                // Forward
                let distance = cpu.get_registers()[2] as f64;
                self.turtle.move_forward(distance);
                Ok(1000)
            }
            2 => {
                // Backward
                let distance = cpu.get_registers()[2] as f64;
                self.turtle.move_backward(distance);
                Ok(1000)
            }
            3 => {
                // Turn left
                let angle = cpu.get_registers()[2] as f64;
                self.turtle.turn_left(angle);
                Ok(500)
            }
            4 => {
                // Turn right
                let angle = cpu.get_registers()[2] as f64;
                self.turtle.turn_right(angle);
                Ok(500)
            }
            5 => {
                // Pen up
                self.turtle.pen_up();
                Ok(200)
            }
            6 => {
                // Pen down
                self.turtle.pen_down();
                Ok(200)
            }
            7 => {
                // Set pen color
                let color = cpu.get_registers()[2];
                self.turtle.set_pen_color(Color::from_hex(color % 0xFFFFFF));
                Ok(300)
            }
            8 => {
                // Set pen size
                let size = cpu.get_registers()[2] as f32;
                self.turtle.set_pen_size(size);
                Ok(300)
            }
            9 => {
                // Clear screen
                self.turtle.clear_screen();
                Ok(5000)
            }
            10 => {
                // Reset
                self.turtle.reset();
                Ok(5000)
            }
            11 => {
                self.turtle
                    .set_speed(cpu.get_registers()[2].clamp(0, TICKS_PER_SECOND));
                Ok(5000)
            }
            _ => Err(hecate_common::ExecutionError::InvalidSyscall(code)),
        }
    }
}

#[derive(Parser)]
struct Args {
    #[command(subcommand)]
    action: Action,
}

#[derive(Subcommand)]
enum Action {
    Run { path: PathBuf },
    RunAsm { path: PathBuf },
}

// Draw the turtle
fn draw_turtle(turtle: &TurtleState) {
    // Turtle's body (circle)
    draw_circle(
        turtle.x as f32,
        turtle.y as f32,
        10.0,  // Radius of the turtle body
        GREEN, // Color of the turtle
    );

    // Turtle's head (line extending from the body in the direction of heading)
    let head_x = turtle.x + 15.0 * turtle.heading.to_radians().cos();
    let head_y = turtle.y - 15.0 * turtle.heading.to_radians().sin();
    draw_line(
        turtle.x as f32,
        turtle.y as f32,
        head_x as f32,
        head_y as f32,
        2.0,
        DARKGREEN,
    );

    // Turtle's legs (4 lines)
    for &angle_offset in &[45.0, -45.0, 135.0, -135.0] {
        let leg_angle = turtle.heading + angle_offset;
        let leg_x = turtle.x + 8.0 * leg_angle.to_radians().cos();
        let leg_y = turtle.y - 8.0 * leg_angle.to_radians().sin();
        draw_line(
            turtle.x as f32,
            turtle.y as f32,
            leg_x as f32,
            leg_y as f32,
            2.0,
            DARKGREEN,
        );
    }
}

const TICKS_PER_SECOND: u32 = 60; // Target tick rate
const SECONDS_PER_TICK: f64 = 1.0 / TICKS_PER_SECOND as f64; // Duration of each tick in seconds

async fn run(rom: &[u32], entrypoint: u32) -> anyhow::Result<()> {
    let mut cpu = NativeCpu::new(1024 * 1024, 32, TurtleIo::new());
    {
        let regs = cpu.get_mut_registers();
        regs[30] = screen_width() as u32;
        regs[31] = screen_height() as u32;
    }
    cpu.set_entrypoint(entrypoint);

    cpu.load_protected_memory(0, rom);

    let virtual_width = screen_width();
    let virtual_height = screen_height();

    let render_target = render_target_msaa(virtual_width as u32, virtual_height as u32, 4);
    render_target.texture.set_filter(FilterMode::Linear);

    let mut render_target_cam =
        Camera2D::from_display_rect(Rect::new(0., 0., virtual_width, virtual_height));
    render_target_cam.render_target = Some(render_target.clone());

    let mut tick = 0;
    let mut last_tick_time = get_time();

    while !cpu.get_halted() {
        if is_key_down(KeyCode::Escape) {
            break;
        }

        let current_time = get_time();
        let elapsed_time = current_time - last_tick_time;

        if elapsed_time >= SECONDS_PER_TICK {
            let elapsed_ticks = (elapsed_time / SECONDS_PER_TICK) as u32;
            for _ in 0..elapsed_ticks {
                if tick >= TICKS_PER_SECOND - cpu.host_io.as_ref().unwrap().turtle.speed {
                    set_camera(&render_target_cam);
                    _ = cpu.execute(RunMode::RunFor(1));
                    tick = 0;
                }
                tick += 1;
            }
            last_tick_time += elapsed_ticks as f64 * SECONDS_PER_TICK;
        }
        set_default_camera();

        clear_background(BLACK);

        draw_texture_ex(
            &render_target.texture,
            0.0,
            0.0,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(virtual_width, virtual_height)),
                flip_y: true, // Must flip y otherwise 'render_target' will be upside down
                ..Default::default()
            },
        );

        draw_turtle(&cpu.host_io.as_ref().unwrap().turtle);

        next_frame().await;
    }

    Ok(())
}

fn conf() -> Conf {
    Conf {
        window_title: "Hecate Turtle".to_string(),
        window_width: 800,
        window_height: 600,
        ..Default::default()
    }
}

#[macroquad::main(conf)]
async fn main() -> anyhow::Result<()> {
    let Args { action } = Args::parse();

    match action {
        Action::Run { path } => {
            let file = BytecodeFile::load(path).unwrap();

            run(&file.data, file.header.entrypoint).await?;
        }

        Action::RunAsm { path } => {
            let program = std::fs::read_to_string(path)?;
            let mut assembler = hecate_assembler::Assembler::new();
            let file = assembler.assemble_program(&program)?;

            run(&file.data, file.header.entrypoint).await?;
        }
    }

    Ok(())
}
