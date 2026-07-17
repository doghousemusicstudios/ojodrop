//! Compile MilkDrop shader bodies without creating a GPU device.
//!
//! This is intentionally small so corpus failures can be reproduced while a
//! renderer gauntlet owns the machine's GPU:
//! `cargo run -p particle-milkdrop --example compile_preset -- <preset>...`

use std::path::Path;

fn main() {
    let mut failed = false;
    let mut args = std::env::args().skip(1).peekable();
    let dump = args.next_if(|argument| argument == "--dump").is_some();
    for raw_path in args {
        let path = Path::new(&raw_path);
        let result = particle_milkdrop::load_preset_path(path).and_then(|preset| {
            if dump {
                if let Some(warp) = &preset.warp {
                    println!(
                        "--- WARP {} ---\n{}",
                        path.display(),
                        particle_milkdrop::preprocess::hlsl_milk_warp_body_to_naga(warp)
                    );
                }
                if let Some(comp) = &preset.comp {
                    println!(
                        "--- COMP {} ---\n{}",
                        path.display(),
                        particle_milkdrop::preprocess::hlsl_milk_body_to_naga(comp)
                    );
                }
            }
            particle_milkdrop::compile_milkdrop_shader_bodies(&preset)
        });
        match result {
            Ok(_) => println!("OK\t{}", path.display()),
            Err(error) => {
                failed = true;
                println!("FAIL\t{}\t{error}", path.display());
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}
