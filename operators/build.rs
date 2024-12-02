fn main() {
    use build_script_cfg::Cfg;
    use search_ascend_tools::find_ascend_toolkit_home;
    use search_cl_tools::find_opencl;
    use search_cuda_tools::{find_cuda_root, find_nccl_root};
    use search_musa_tools::find_musa_home;
    use std::{env, path::PathBuf};

    let cpu = Cfg::new("use_cpu");
    let cl = Cfg::new("use_cl");
    let cuda = Cfg::new("use_cuda");
    let nccl = Cfg::new("use_nccl");
    let ascend = Cfg::new("use_ascend");
    let musa = Cfg::new("use_musa");

    if cfg!(feature = "common-cpu") {
        cpu.define();
    }
    if cfg!(feature = "opencl") && find_opencl().is_some() {
        cl.define();
    }
    if cfg!(feature = "nvidia-gpu") && find_cuda_root().is_some() {
        cuda.define();
        if find_nccl_root().is_some() {
            nccl.define();
        }
    }
    if cfg!(feature = "ascend") && find_ascend_toolkit_home().is_some() {
        ascend.define();
    }
    if cfg!(feature = "mthreads-gpu") && find_musa_home().is_some() {
        musa.define();
        let infini_root = env::var("INFINI_ROOT").unwrap_or_else(|_| {
            eprintln!("INFINI_ROOT environment variable is not set.");
            std::process::exit(1)
        }) + "lib";

        println!("cargo:rustc-link-search=native={}", infini_root);
        println!("cargo:rustc-link-lib=dylib=infiniop");
        println!("cargo:rerun-if-changed=wrapper.h");

        let bindings = bindgen::Builder::default()
            .header("./src/handle/mthreads_gpu/wrapper.h")
            .clang_arg(format!("-I{}/include", infini_root))
            .clang_arg("-xc++")
            .allowlist_function(".*")
            .allowlist_item(".*")
            .must_use_type("infiniopStatus_t")
            .default_enum_style(bindgen::EnumVariation::Rust {
                non_exhaustive: true,
            })
            .use_core()
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate bindings");

        // Write the bindings to the $OUT_DIR/bindings.rs file.

        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        println!("{}", out_path.display());
        bindings
            .write_to_file(out_path.join("opbindings.rs"))
            .expect("Couldn't write bindings!");
    }
}
